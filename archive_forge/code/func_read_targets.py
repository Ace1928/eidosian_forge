from flask import Flask, jsonify, send_from_directory
from glob import glob
@app.route('/')
def read_targets():
    return jsonify([el.replace('./char-dataset', '/raw').replace('\\', '/') for el in sorted(glob('./char-dataset/**/*'))])