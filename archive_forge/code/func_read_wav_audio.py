from __future__ import division
import array
import os
import subprocess
from tempfile import TemporaryFile, NamedTemporaryFile
import wave
import sys
import struct
from .logging_utils import log_conversion, log_subprocess_output
from .utils import mediainfo_json, fsdecode
import base64
from collections import namedtuple
from io import BytesIO
from .utils import (
from .exceptions import (
from . import effects
def read_wav_audio(data, headers=None):
    if not headers:
        headers = extract_wav_headers(data)
    fmt = [x for x in headers if x.id == b'fmt ']
    if not fmt or fmt[0].size < 16:
        raise CouldntDecodeError("Couldn't find fmt header in wav data")
    fmt = fmt[0]
    pos = fmt.position + 8
    audio_format = struct.unpack_from('<H', data[pos:pos + 2])[0]
    if audio_format != 1 and audio_format != 65534:
        raise CouldntDecodeError('Unknown audio format 0x%X in wav data' % audio_format)
    channels = struct.unpack_from('<H', data[pos + 2:pos + 4])[0]
    sample_rate = struct.unpack_from('<I', data[pos + 4:pos + 8])[0]
    bits_per_sample = struct.unpack_from('<H', data[pos + 14:pos + 16])[0]
    data_hdr = headers[-1]
    if data_hdr.id != b'data':
        raise CouldntDecodeError("Couldn't find data header in wav data")
    pos = data_hdr.position + 8
    return WavData(audio_format, channels, sample_rate, bits_per_sample, data[pos:pos + data_hdr.size])