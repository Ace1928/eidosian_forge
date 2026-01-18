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
@classmethod
def from_mono_audiosegments(cls, *mono_segments):
    if not len(mono_segments):
        raise ValueError('At least one AudioSegment instance is required')
    segs = cls._sync(*mono_segments)
    if segs[0].channels != 1:
        raise ValueError('AudioSegment.from_mono_audiosegments requires all arguments are mono AudioSegment instances')
    channels = len(segs)
    sample_width = segs[0].sample_width
    frame_rate = segs[0].frame_rate
    frame_count = max((int(seg.frame_count()) for seg in segs))
    data = array.array(segs[0].array_type, b'\x00' * (frame_count * sample_width * channels))
    for i, seg in enumerate(segs):
        data[i::channels] = seg.get_array_of_samples()
    return cls(data, channels=channels, sample_width=sample_width, frame_rate=frame_rate)