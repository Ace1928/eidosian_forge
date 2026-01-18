import langchain
from langchain.llms import Replicate
from flask import Flask
from flask import request
import os
import requests
import json
@app.route('/')
def hello_llama():
    return '<p>Hello Llama 2</p>'