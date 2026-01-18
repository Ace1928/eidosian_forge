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
def ffmpeg_open_stream(file, index):
    if not 0 <= index < file.context.contents.nb_streams:
        raise FFmpegException('index out of range. Only {} streams.'.format(file.context.contents.nb_streams))
    codec_context = avcodec.avcodec_alloc_context3(None)
    if not codec_context:
        raise MemoryError('Could not allocate Codec Context.')
    result = avcodec.avcodec_parameters_to_context(codec_context, file.context.contents.streams[index].contents.codecpar)
    if result < 0:
        avcodec.avcodec_free_context(byref(codec_context))
        raise FFmpegException('Could not copy the AVCodecContext.')
    codec_id = codec_context.contents.codec_id
    codec = avcodec.avcodec_find_decoder(codec_id)
    if _debug:
        print('Found Codec=', codec_id, '=', codec.contents.long_name.decode())
    if codec_id == AV_CODEC_ID_VP9:
        newcodec = avcodec.avcodec_find_decoder_by_name('libvpx-vp9'.encode('utf-8'))
        codec = newcodec or codec
    if codec_id == AV_CODEC_ID_VP8:
        newcodec = avcodec.avcodec_find_decoder_by_name('libvpx'.encode('utf-8'))
        codec = newcodec or codec
    if not codec:
        raise FFmpegException('No codec found for this media. codecID={}'.format(codec_id))
    codec_id = codec.contents.id
    if _debug:
        print('Loaded codec: ', codec.contents.long_name.decode())
    result = avcodec.avcodec_open2(codec_context, codec, None)
    if result < 0:
        raise FFmpegException('Could not open the media with the codec.')
    stream = FFmpegStream()
    stream.format_context = file.context
    stream.codec_context = codec_context
    stream.type = codec_context.contents.codec_type
    stream.frame = avutil.av_frame_alloc()
    stream.time_base = file.context.contents.streams[index].contents.time_base
    return stream