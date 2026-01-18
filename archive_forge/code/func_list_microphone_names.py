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
@staticmethod
def list_microphone_names():
    """
        Returns a list of the names of all available microphones. For microphones where the name can't be retrieved, the list entry contains ``None`` instead.

        The index of each microphone's name in the returned list is the same as its device index when creating a ``Microphone`` instance - if you want to use the microphone at index 3 in the returned list, use ``Microphone(device_index=3)``.
        """
    audio = Microphone.get_pyaudio().PyAudio()
    try:
        result = []
        for i in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(i)
            result.append(device_info.get('name'))
    finally:
        audio.terminate()
    return result