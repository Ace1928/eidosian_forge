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
def set_frame_rate(self, frame_rate):
    if frame_rate == self.frame_rate:
        return self
    if self._data:
        converted, _ = audioop.ratecv(self._data, self.sample_width, self.channels, self.frame_rate, frame_rate, None)
    else:
        converted = self._data
    return self._spawn(data=converted, overrides={'frame_rate': frame_rate})