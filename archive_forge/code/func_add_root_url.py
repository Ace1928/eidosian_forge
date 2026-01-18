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
def add_root_url(data: dict | list, root_url: str, previous_root_url: str | None):

    def _add_root_url(file_dict: dict):
        if previous_root_url and file_dict['url'].startswith(previous_root_url):
            file_dict['url'] = file_dict['url'][len(previous_root_url):]
        elif client_utils.is_http_url_like(file_dict['url']):
            return file_dict
        file_dict['url'] = f'{root_url}{file_dict['url']}'
        return file_dict
    return client_utils.traverse(data, _add_root_url, client_utils.is_file_obj_with_url)