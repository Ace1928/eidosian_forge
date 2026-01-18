from flask import Flask, jsonify, send_from_directory
from glob import glob
@app.route('/edge/gb18030/json')
def read_gb18030_response_json():
    return ('{"abc": "我没有埋怨，磋砣的只是一些时间。。今觀俗士之論也，以族舉德，以位命賢，茲可謂得論之一體矣，而未獲至論之淑真也。"}'.encode('gb18030'), 200, {'Content-Type': 'application/json'})