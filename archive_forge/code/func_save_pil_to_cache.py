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
def save_pil_to_cache(img: Image.Image, cache_dir: str, name: str='image', format: str='webp') -> str:
    bytes_data = encode_pil_to_bytes(img, format)
    temp_dir = Path(cache_dir) / hash_bytes(bytes_data)
    temp_dir.mkdir(exist_ok=True, parents=True)
    filename = str((temp_dir / f'{name}.{format}').resolve())
    (temp_dir / f'{name}.{format}').resolve().write_bytes(bytes_data)
    return filename