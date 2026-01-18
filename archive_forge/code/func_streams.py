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
def streams():
    format_context = self._file.context
    for idx in (self._video_stream_index, self._audio_stream_index):
        if idx is None:
            continue
        stream = format_context.contents.streams[idx].contents
        yield stream