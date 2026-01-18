from __future__ import annotations
import base64
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import warnings
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
import aiofiles
import httpx
import numpy as np
from gradio_client import utils as client_utils
from PIL import Image, ImageOps, PngImagePlugin
from gradio import utils, wasm_utils
from gradio.data_classes import FileData, GradioModel, GradioRootModel
from gradio.utils import abspath, get_upload_folder, is_in_or_equal
def audio_from_file(filename, crop_min=0, crop_max=100):
    try:
        audio = AudioSegment.from_file(filename)
    except FileNotFoundError as e:
        isfile = Path(filename).is_file()
        msg = f'Cannot load audio from file: `{('ffprobe' if isfile else filename)}` not found.' + ' Please install `ffmpeg` in your system to use non-WAV audio file formats and make sure `ffprobe` is in your PATH.' if isfile else ''
        raise RuntimeError(msg) from e
    if crop_min != 0 or crop_max != 100:
        audio_start = len(audio) * crop_min / 100
        audio_end = len(audio) * crop_max / 100
        audio = audio[audio_start:audio_end]
    data = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        data = data.reshape(-1, audio.channels)
    return (audio.frame_rate, data)