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
def recognize_azure(self, audio_data, key, language='en-US', profanity='masked', location='westus', show_all=False):
    """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using the Microsoft Azure Speech API.

        The Microsoft Azure Speech API key is specified by ``key``. Unfortunately, these are not available without `signing up for an account <https://azure.microsoft.com/en-ca/pricing/details/cognitive-services/speech-api/>`__ with Microsoft Azure.

        To get the API key, go to the `Microsoft Azure Portal Resources <https://portal.azure.com/>`__ page, go to "All Resources" > "Add" > "See All" > Search "Speech > "Create", and fill in the form to make a "Speech" resource. On the resulting page (which is also accessible from the "All Resources" page in the Azure Portal), go to the "Show Access Keys" page, which will have two API keys, either of which can be used for the `key` parameter. Microsoft Azure Speech API keys are 32-character lowercase hexadecimal strings.

        The recognition language is determined by ``language``, a BCP-47 language tag like ``"en-US"`` (US English) or ``"fr-FR"`` (International French), defaulting to US English. A list of supported language values can be found in the `API documentation <https://docs.microsoft.com/en-us/azure/cognitive-services/speech/api-reference-rest/bingvoicerecognition#recognition-language>`__ under "Interactive and dictation mode".

        Returns the most likely transcription if ``show_all`` is false (the default). Otherwise, returns the `raw API response <https://docs.microsoft.com/en-us/azure/cognitive-services/speech/api-reference-rest/bingvoicerecognition#sample-responses>`__ as a JSON dictionary.

        Raises a ``speech_recognition.UnknownValueError`` exception if the speech is unintelligible. Raises a ``speech_recognition.RequestError`` exception if the speech recognition operation failed, if the key isn't valid, or if there is no internet connection.
        """
    assert isinstance(audio_data, AudioData), 'Data must be audio data'
    assert isinstance(key, str), '``key`` must be a string'
    assert isinstance(language, str), '``language`` must be a string'
    result_format = 'detailed'
    access_token, expire_time = (getattr(self, 'azure_cached_access_token', None), getattr(self, 'azure_cached_access_token_expiry', None))
    allow_caching = True
    try:
        from time import monotonic
    except ImportError:
        expire_time = None
        allow_caching = False
    if expire_time is None or monotonic() > expire_time:
        credential_url = 'https://' + location + '.api.cognitive.microsoft.com/sts/v1.0/issueToken'
        credential_request = Request(credential_url, data=b'', headers={'Content-type': 'application/x-www-form-urlencoded', 'Content-Length': '0', 'Ocp-Apim-Subscription-Key': key})
        if allow_caching:
            start_time = monotonic()
        try:
            credential_response = urlopen(credential_request, timeout=60)
        except HTTPError as e:
            raise RequestError('credential request failed: {}'.format(e.reason))
        except URLError as e:
            raise RequestError('credential connection failed: {}'.format(e.reason))
        access_token = credential_response.read().decode('utf-8')
        if allow_caching:
            self.azure_cached_access_token = access_token
            self.azure_cached_access_token_expiry = start_time + 600
    wav_data = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
    url = 'https://' + location + '.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?{}'.format(urlencode({'language': language, 'format': result_format, 'profanity': profanity}))
    if sys.version_info >= (3, 6):
        request = Request(url, data=io.BytesIO(wav_data), headers={'Authorization': 'Bearer {}'.format(access_token), 'Content-type': 'audio/wav; codec="audio/pcm"; samplerate=16000', 'Transfer-Encoding': 'chunked'})
    else:
        ascii_hex_data_length = '{:X}'.format(len(wav_data)).encode('utf-8')
        chunked_transfer_encoding_data = ascii_hex_data_length + b'\r\n' + wav_data + b'\r\n0\r\n\r\n'
        request = Request(url, data=chunked_transfer_encoding_data, headers={'Authorization': 'Bearer {}'.format(access_token), 'Content-type': 'audio/wav; codec="audio/pcm"; samplerate=16000', 'Transfer-Encoding': 'chunked'})
    try:
        response = urlopen(request, timeout=self.operation_timeout)
    except HTTPError as e:
        raise RequestError('recognition request failed: {}'.format(e.reason))
    except URLError as e:
        raise RequestError('recognition connection failed: {}'.format(e.reason))
    response_text = response.read().decode('utf-8')
    result = json.loads(response_text)
    if show_all:
        return result
    if 'RecognitionStatus' not in result or result['RecognitionStatus'] != 'Success' or 'NBest' not in result:
        raise UnknownValueError()
    return (result['NBest'][0]['Display'], result['NBest'][0]['Confidence'])