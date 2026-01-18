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
def list_working_microphones():
    """
        Returns a dictionary mapping device indices to microphone names, for microphones that are currently hearing sounds. When using this function, ensure that your microphone is unmuted and make some noise at it to ensure it will be detected as working.

        Each key in the returned dictionary can be passed to the ``Microphone`` constructor to use that microphone. For example, if the return value is ``{3: "HDA Intel PCH: ALC3232 Analog (hw:1,0)"}``, you can do ``Microphone(device_index=3)`` to use that microphone.
        """
    pyaudio_module = Microphone.get_pyaudio()
    audio = pyaudio_module.PyAudio()
    try:
        result = {}
        for device_index in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(device_index)
            device_name = device_info.get('name')
            assert isinstance(device_info.get('defaultSampleRate'), (float, int)) and device_info['defaultSampleRate'] > 0, 'Invalid device info returned from PyAudio: {}'.format(device_info)
            try:
                pyaudio_stream = audio.open(input_device_index=device_index, channels=1, format=pyaudio_module.paInt16, rate=int(device_info['defaultSampleRate']), input=True)
                try:
                    buffer = pyaudio_stream.read(1024)
                    if not pyaudio_stream.is_stopped():
                        pyaudio_stream.stop_stream()
                finally:
                    pyaudio_stream.close()
            except Exception:
                continue
            energy = -audioop.rms(buffer, 2)
            energy_bytes = bytes([energy & 255, energy >> 8 & 255])
            debiased_energy = audioop.rms(audioop.add(buffer, energy_bytes * (len(buffer) // 2), 2), 2)
            if debiased_energy > 30:
                result[device_index] = device_name
    finally:
        audio.terminate()
    return result