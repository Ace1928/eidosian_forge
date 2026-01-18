from flask import Flask, jsonify, send_from_directory
from glob import glob
@app.route('/edge/empty/json')
def read_empty_response_json():
    return (b'{}', 200, {'Content-Type': 'application/json'})