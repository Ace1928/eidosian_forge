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
def recognize_lex(self, audio_data, bot_name, bot_alias, user_id, content_type='audio/l16; rate=16000; channels=1', access_key_id=None, secret_access_key=None, region=None):
    """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using the Amazon Lex API.

        If access_key_id or secret_access_key is not set it will go through the list in the link below
        http://boto3.readthedocs.io/en/latest/guide/configuration.html#configuring-credentials
        """
    assert isinstance(audio_data, AudioData), 'Data must be audio data'
    assert isinstance(bot_name, str), '``bot_name`` must be a string'
    assert isinstance(bot_alias, str), '``bot_alias`` must be a string'
    assert isinstance(user_id, str), '``user_id`` must be a string'
    assert isinstance(content_type, str), '``content_type`` must be a string'
    assert access_key_id is None or isinstance(access_key_id, str), '``access_key_id`` must be a string'
    assert secret_access_key is None or isinstance(secret_access_key, str), '``secret_access_key`` must be a string'
    assert region is None or isinstance(region, str), '``region`` must be a string'
    try:
        import boto3
    except ImportError:
        raise RequestError('missing boto3 module: ensure that boto3 is set up correctly.')
    client = boto3.client('lex-runtime', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key, region_name=region)
    raw_data = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
    accept = 'text/plain; charset=utf-8'
    response = client.post_content(botName=bot_name, botAlias=bot_alias, userId=user_id, contentType=content_type, accept=accept, inputStream=raw_data)
    return response['inputTranscript']