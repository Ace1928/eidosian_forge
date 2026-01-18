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
def hash_base64(base64_encoding: str, chunk_num_blocks: int=128) -> str:
    sha1 = hashlib.sha1()
    for i in range(0, len(base64_encoding), chunk_num_blocks * sha1.block_size):
        data = base64_encoding[i:i + chunk_num_blocks * sha1.block_size]
        sha1.update(data.encode('utf-8'))
    return sha1.hexdigest()