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
def save_base64_to_cache(base64_encoding: str, cache_dir: str, file_name: str | None=None) -> str:
    """Converts a base64 encoding to a file and returns the path to the file if
    the file doesn't already exist. Otherwise returns the path to the existing file.
    """
    temp_dir = hash_base64(base64_encoding)
    temp_dir = Path(cache_dir) / temp_dir
    temp_dir.mkdir(exist_ok=True, parents=True)
    guess_extension = client_utils.get_extension(base64_encoding)
    if file_name:
        file_name = client_utils.strip_invalid_filename_characters(file_name)
    elif guess_extension:
        file_name = f'file.{guess_extension}'
    else:
        file_name = 'file'
    full_temp_file_path = str(abspath(temp_dir / file_name))
    if not Path(full_temp_file_path).exists():
        data, _ = client_utils.decode_base64_to_binary(base64_encoding)
        with open(full_temp_file_path, 'wb') as fb:
            fb.write(data)
    return full_temp_file_path