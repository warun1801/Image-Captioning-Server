import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import ImgCap
from tensorflow.keras.models import load_model
from pickle import load

model = load_model('model-ep005-loss3.539-val_loss3.842.h5')
tokenizer = load(open('tokenizer.pkl', 'rb'))

UPLOAD_FOLDER = os.getcwd()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            print('File Saved', os.path.join(
                app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('showCaption',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/result', methods=['GET', 'POST'],)
def showCaption():
    if request.method == 'GET':
        filename = request.args.get('filename')
        desc = ImgCap.get_caption(filename, model, tokenizer, 34)

        return desc

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
