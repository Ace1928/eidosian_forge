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
def ffmpeg_stream_info(file, stream_index):
    """Open the stream
    """
    av_stream = file.context.contents.streams[stream_index].contents
    context = av_stream.codecpar.contents
    if context.codec_type == AVMEDIA_TYPE_VIDEO:
        if _debug:
            print('codec_type=', context.codec_type)
            print(' codec_id=', context.codec_id)
            print(' codec name=', avcodec.avcodec_get_name(context.codec_id).decode('utf-8'))
            print(' codec_tag=', context.codec_tag)
            print(' extradata=', context.extradata)
            print(' extradata_size=', context.extradata_size)
            print(' format=', context.format)
            print(' bit_rate=', context.bit_rate)
            print(' bits_per_coded_sample=', context.bits_per_coded_sample)
            print(' bits_per_raw_sample=', context.bits_per_raw_sample)
            print(' profile=', context.profile)
            print(' level=', context.level)
            print(' width=', context.width)
            print(' height=', context.height)
            print(' sample_aspect_ratio=', context.sample_aspect_ratio.num, context.sample_aspect_ratio.den)
            print(' field_order=', context.field_order)
            print(' color_range=', context.color_range)
            print(' color_primaries=', context.color_primaries)
            print(' color_trc=', context.color_trc)
            print(' color_space=', context.color_space)
            print(' chroma_location=', context.chroma_location)
            print(' video_delay=', context.video_delay)
            print(' channel_layout=', context.channel_layout)
            print(' channels=', context.channels)
            print(' sample_rate=', context.sample_rate)
            print(' block_align=', context.block_align)
            print(' frame_size=', context.frame_size)
            print(' initial_padding=', context.initial_padding)
            print(' trailing_padding=', context.trailing_padding)
            print(' seek_preroll=', context.seek_preroll)
        frame_rate = avformat.av_guess_frame_rate(file.context, av_stream, None)
        info = StreamVideoInfo(context.width, context.height, context.sample_aspect_ratio.num, context.sample_aspect_ratio.den, frame_rate.num, frame_rate.den, context.codec_id)
    elif context.codec_type == AVMEDIA_TYPE_AUDIO:
        info = StreamAudioInfo(context.format, context.sample_rate, context.channels)
        if context.format in (AV_SAMPLE_FMT_U8, AV_SAMPLE_FMT_U8P):
            info.sample_bits = 8
        elif context.format in (AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_S16P, AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_FLTP):
            info.sample_bits = 16
        elif context.format in (AV_SAMPLE_FMT_S32, AV_SAMPLE_FMT_S32P):
            info.sample_bits = 32
        else:
            info.sample_format = None
            info.sample_bits = None
    else:
        return None
    return info