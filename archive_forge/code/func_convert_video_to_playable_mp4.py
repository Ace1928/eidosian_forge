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
def convert_video_to_playable_mp4(video_path: str) -> str:
    """Convert the video to mp4. If something goes wrong return the original video."""
    from ffmpy import FFmpeg, FFRuntimeError
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            output_path = Path(video_path).with_suffix('.mp4')
            shutil.copy2(video_path, tmp_file.name)
            ff = FFmpeg(inputs={str(tmp_file.name): None}, outputs={str(output_path): None}, global_options='-y -loglevel quiet')
            ff.run()
    except FFRuntimeError as e:
        print(f'Error converting video to browser-playable format {str(e)}')
        output_path = video_path
    finally:
        os.remove(tmp_file.name)
    return str(output_path)