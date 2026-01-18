from __future__ import annotations
import aifc
import audioop
import base64
import collections
import hashlib
import hmac
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from .audio import AudioData, get_flac_converter
from .exceptions import (
def recognize_assemblyai(self, audio_data, api_token, job_name=None, **kwargs):
    """
        Wraps the AssemblyAI STT service.
        https://www.assemblyai.com/
        """

    def read_file(filename, chunk_size=5242880):
        with open(filename, 'rb') as _file:
            while True:
                data = _file.read(chunk_size)
                if not data:
                    break
                yield data
    check_existing = audio_data is None and job_name
    if check_existing:
        transciption_id = job_name
        endpoint = f'https://api.assemblyai.com/v2/transcript/{transciption_id}'
        headers = {'authorization': api_token}
        response = requests.get(endpoint, headers=headers)
        data = response.json()
        status = data['status']
        if status == 'error':
            exc = TranscriptionFailed()
            exc.job_name = None
            exc.file_key = None
            raise exc
        elif status == 'completed':
            confidence = data['confidence']
            text = data['text']
            return (text, confidence)
        print('Keep waiting.')
        exc = TranscriptionNotReady()
        exc.job_name = job_name
        exc.file_key = None
        raise exc
    else:
        headers = {'authorization': api_token}
        response = requests.post('https://api.assemblyai.com/v2/upload', headers=headers, data=read_file(audio_data))
        upload_url = response.json()['upload_url']
        endpoint = 'https://api.assemblyai.com/v2/transcript'
        json = {'audio_url': upload_url}
        headers = {'authorization': api_token, 'content-type': 'application/json'}
        response = requests.post(endpoint, json=json, headers=headers)
        data = response.json()
        transciption_id = data['id']
        exc = TranscriptionNotReady()
        exc.job_name = transciption_id
        exc.file_key = None
        raise exc