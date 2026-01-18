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
def split_to_mono(self):
    if self.channels == 1:
        return [self]
    samples = self.get_array_of_samples()
    mono_channels = []
    for i in range(self.channels):
        samples_for_current_channel = samples[i::self.channels]
        try:
            mono_data = samples_for_current_channel.tobytes()
        except AttributeError:
            mono_data = samples_for_current_channel.tostring()
        mono_channels.append(self._spawn(mono_data, overrides={'channels': 1, 'frame_width': self.sample_width}))
    return mono_channels