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
def ffmpeg_seek_file(file, timestamp):
    flags = 0
    max_ts = file.context.contents.duration * AV_TIME_BASE
    result = avformat.avformat_seek_file(file.context, -1, 0, timestamp, timestamp, flags)
    if result < 0:
        buf = create_string_buffer(128)
        avutil.av_strerror(result, buf, 128)
        descr = buf.value
        raise FFmpegException('Error occured while seeking. ' + descr.decode())