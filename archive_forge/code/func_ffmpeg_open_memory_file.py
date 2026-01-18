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
def ffmpeg_open_memory_file(filename, file_object):
    """Open a media file from a file object.
    :rtype: FFmpegFile
    :return: The structure containing all the information for the media.
    """
    file = FFmpegFile()
    file.context = libavformat.avformat.avformat_alloc_context()
    file.context.contents.seekable = 1
    memory_file = MemoryFileObject(file_object)
    av_buf = libavutil.avutil.av_malloc(memory_file.buffer_size)
    memory_file.buffer = cast(av_buf, c_char_p)
    ptr = create_string_buffer(memory_file.buffer_size)
    memory_file.fmt_context = libavformat.avformat.avio_alloc_context(memory_file.buffer, memory_file.buffer_size, 0, ptr, memory_file.read_func, None, memory_file.seek_func)
    file.context.contents.pb = memory_file.fmt_context
    file.context.contents.flags |= libavformat.AVFMT_FLAG_CUSTOM_IO
    result = avformat.avformat_open_input(byref(file.context), filename, None, None)
    if result != 0:
        raise FFmpegException('avformat_open_input in ffmpeg_open_filename returned an error opening file ' + filename.decode('utf8') + ' Error code: ' + str(result))
    result = avformat.avformat_find_stream_info(file.context, None)
    if result < 0:
        raise FFmpegException('Could not find stream info')
    return (file, memory_file)