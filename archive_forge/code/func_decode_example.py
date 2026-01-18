import os
import sys
import warnings
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union
import numpy as np
import pyarrow as pa
from .. import config
from ..download.download_config import DownloadConfig
from ..download.streaming_download_manager import xopen
from ..table import array_cast
from ..utils.file_utils import is_local_path
from ..utils.py_utils import first_non_null_value, no_op_if_value_is_null, string_to_dict
def decode_example(self, value: dict, token_per_repo_id=None) -> 'PIL.Image.Image':
    """Decode example image file into image data.

        Args:
            value (`str` or `dict`):
                A string with the absolute image file path, a dictionary with
                keys:

                - `path`: String with absolute or relative image file path.
                - `bytes`: The bytes of the image file.
            token_per_repo_id (`dict`, *optional*):
                To access and decode
                image files from private repositories on the Hub, you can pass
                a dictionary repo_id (`str`) -> token (`bool` or `str`).

        Returns:
            `PIL.Image.Image`
        """
    if not self.decode:
        raise RuntimeError('Decoding is disabled for this feature. Please use Image(decode=True) instead.')
    if config.PIL_AVAILABLE:
        import PIL.Image
    else:
        raise ImportError("To support decoding images, please install 'Pillow'.")
    if token_per_repo_id is None:
        token_per_repo_id = {}
    path, bytes_ = (value['path'], value['bytes'])
    if bytes_ is None:
        if path is None:
            raise ValueError(f"An image should have one of 'path' or 'bytes' but both are None in {value}.")
        elif is_local_path(path):
            image = PIL.Image.open(path)
        else:
            source_url = path.split('::')[-1]
            pattern = config.HUB_DATASETS_URL if source_url.startswith(config.HF_ENDPOINT) else config.HUB_DATASETS_HFFS_URL
            try:
                repo_id = string_to_dict(source_url, pattern)['repo_id']
                token = token_per_repo_id.get(repo_id)
            except ValueError:
                token = None
            download_config = DownloadConfig(token=token)
            with xopen(path, 'rb', download_config=download_config) as f:
                bytes_ = BytesIO(f.read())
            image = PIL.Image.open(bytes_)
    else:
        image = PIL.Image.open(BytesIO(bytes_))
    image.load()
    return image