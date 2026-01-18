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
def save_bytes_to_cache(data: bytes, file_name: str, cache_dir: str) -> str:
    path = Path(cache_dir) / hash_bytes(data)
    path.mkdir(exist_ok=True, parents=True)
    path = path / Path(file_name).name
    path.write_bytes(data)
    return str(path.resolve())