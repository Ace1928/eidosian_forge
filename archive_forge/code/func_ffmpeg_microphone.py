import datetime
import platform
import subprocess
from typing import Optional, Tuple, Union
import numpy as np
def ffmpeg_microphone(sampling_rate: int, chunk_length_s: float, format_for_conversion: str='f32le'):
    """
    Helper function to read raw microphone data.
    """
    ar = f'{sampling_rate}'
    ac = '1'
    if format_for_conversion == 's16le':
        size_of_sample = 2
    elif format_for_conversion == 'f32le':
        size_of_sample = 4
    else:
        raise ValueError(f'Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`')
    system = platform.system()
    if system == 'Linux':
        format_ = 'alsa'
        input_ = 'default'
    elif system == 'Darwin':
        format_ = 'avfoundation'
        input_ = ':0'
    elif system == 'Windows':
        format_ = 'dshow'
        input_ = _get_microphone_name()
    ffmpeg_command = ['ffmpeg', '-f', format_, '-i', input_, '-ac', ac, '-ar', ar, '-f', format_for_conversion, '-fflags', 'nobuffer', '-hide_banner', '-loglevel', 'quiet', 'pipe:1']
    chunk_len = int(round(sampling_rate * chunk_length_s)) * size_of_sample
    iterator = _ffmpeg_stream(ffmpeg_command, chunk_len)
    for item in iterator:
        yield item