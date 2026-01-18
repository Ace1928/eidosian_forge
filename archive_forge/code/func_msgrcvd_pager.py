import langchain
from langchain.llms import Replicate
from flask import Flask
from flask import request
import os
import requests
import json
@app.route('/msgrcvd_pager', methods=['POST', 'GET'])
def msgrcvd_pager():
    message = request.args.get('message')
    sender = request.args.get('sender')
    recipient = request.args.get('recipient')
    answer = llm(message)
    print(message)
    print(answer)
    url = f'https://graph.facebook.com/v18.0/{recipient}/messages'
    params = {'recipient': '{"id": ' + sender + '}', 'message': json.dumps({'text': answer}), 'messaging_type': 'RESPONSE', 'access_token': '<your page access token>'}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, params=params, headers=headers)
    print(response.status_code)
    print(response.text)
    return message + '<p/>' + answer