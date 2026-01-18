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
def ffmpeg_file_info(file):
    """Get information on the file:

        - number of streams
        - duration
        - artist
        - album
        - date
        - track

    :rtype: FileInfo
    :return: The file info instance containing all the meta information.
    """
    info = FileInfo()
    info.n_streams = file.context.contents.nb_streams
    info.start_time = file.context.contents.start_time
    info.duration = file.context.contents.duration
    entry = avutil.av_dict_get(file.context.contents.metadata, asbytes('title'), None, 0)
    if entry:
        info.title = asstr(entry.contents.value)
    entry = avutil.av_dict_get(file.context.contents.metadata, asbytes('artist'), None, 0) or avutil.av_dict_get(file.context.contents.metadata, asbytes('album_artist'), None, 0)
    if entry:
        info.author = asstr(entry.contents.value)
    entry = avutil.av_dict_get(file.context.contents.metadata, asbytes('copyright'), None, 0)
    if entry:
        info.copyright = asstr(entry.contents.value)
    entry = avutil.av_dict_get(file.context.contents.metadata, asbytes('comment'), None, 0)
    if entry:
        info.comment = asstr(entry.contents.value)
    entry = avutil.av_dict_get(file.context.contents.metadata, asbytes('album'), None, 0)
    if entry:
        info.album = asstr(entry.contents.value)
    entry = avutil.av_dict_get(file.context.contents.metadata, asbytes('date'), None, 0)
    if entry:
        info.year = asstr(entry.contents.value)
    entry = avutil.av_dict_get(file.context.contents.metadata, asbytes('track'), None, 0)
    if entry:
        info.track = asstr(entry.contents.value)
    entry = avutil.av_dict_get(file.context.contents.metadata, asbytes('genre'), None, 0)
    if entry:
        info.genre = asstr(entry.contents.value)
    return info