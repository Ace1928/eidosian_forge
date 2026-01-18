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
@property
def max_possible_amplitude(self):
    bits = self.sample_width * 8
    max_possible_val = 2 ** bits
    return max_possible_val / 2