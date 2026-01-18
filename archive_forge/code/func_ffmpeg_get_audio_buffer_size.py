import sys
from collections import deque
from ctypes import (c_int, c_int32, c_uint8, c_char_p,
import pyglet
import pyglet.lib
from pyglet import image
from pyglet.util import asbytes, asstr
from . import MediaDecoder
from .base import AudioData, SourceInfo, StaticSource
from .base import StreamingSource, VideoFormat, AudioFormat
from .ffmpeg_lib import *
from ..exceptions import MediaFormatException
def ffmpeg_get_audio_buffer_size(audio_format):
    """Return the audio buffer size

    Buffer size can accomodate 1 sec of audio data.
    """
    return audio_format.bytes_per_second + FF_INPUT_BUFFER_PADDING_SIZE