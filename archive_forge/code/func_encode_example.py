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
def encode_example(self, value: Union[str, bytes, dict, np.ndarray, 'PIL.Image.Image']) -> dict:
    """Encode example into a format for Arrow.

        Args:
            value (`str`, `np.ndarray`, `PIL.Image.Image` or `dict`):
                Data passed as input to Image feature.

        Returns:
            `dict` with "path" and "bytes" fields
        """
    if config.PIL_AVAILABLE:
        import PIL.Image
    else:
        raise ImportError("To support encoding images, please install 'Pillow'.")
    if isinstance(value, list):
        value = np.array(value)
    if isinstance(value, str):
        return {'path': value, 'bytes': None}
    elif isinstance(value, bytes):
        return {'path': None, 'bytes': value}
    elif isinstance(value, np.ndarray):
        return encode_np_array(value)
    elif isinstance(value, PIL.Image.Image):
        return encode_pil_image(value)
    elif value.get('path') is not None and os.path.isfile(value['path']):
        return {'bytes': None, 'path': value.get('path')}
    elif value.get('bytes') is not None or value.get('path') is not None:
        return {'bytes': value.get('bytes'), 'path': value.get('path')}
    else:
        raise ValueError(f"An image sample should have one of 'path' or 'bytes' but they are missing or None in {value}.")