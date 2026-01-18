import subprocess
from typing import Union
import numpy as np
import requests
from ..utils import add_end_docstrings, is_torch_available, is_torchaudio_available, logging
from .base import Pipeline, build_pipeline_init_args
def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    Helper function to read an audio file through ffmpeg.
    """
    ar = f'{sampling_rate}'
    ac = '1'
    format_for_conversion = 'f32le'
    ffmpeg_command = ['ffmpeg', '-i', 'pipe:0', '-ac', ac, '-ar', ar, '-f', format_for_conversion, '-hide_banner', '-loglevel', 'quiet', 'pipe:1']
    try:
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    except FileNotFoundError:
        raise ValueError('ffmpeg was not found but is required to load audio files from filename')
    output_stream = ffmpeg_process.communicate(bpayload)
    out_bytes = output_stream[0]
    audio = np.frombuffer(out_bytes, np.float32)
    if audio.shape[0] == 0:
        raise ValueError('Malformed soundfile')
    return audio