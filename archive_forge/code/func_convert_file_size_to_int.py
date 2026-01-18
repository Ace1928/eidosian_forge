import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import warnings
from concurrent import futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import uuid4
import huggingface_hub
import requests
from huggingface_hub import (
from huggingface_hub.file_download import REGEX_COMMIT_HASH, http_get
from huggingface_hub.utils import (
from huggingface_hub.utils._deprecation import _deprecate_method
from requests.exceptions import HTTPError
from . import __version__, logging
from .generic import working_or_temp_dir
from .import_utils import (
from .logging import tqdm
def convert_file_size_to_int(size: Union[int, str]):
    """
    Converts a size expressed as a string with digits an unit (like `"5MB"`) to an integer (in bytes).

    Args:
        size (`int` or `str`): The size to convert. Will be directly returned if an `int`.

    Example:
    ```py
    >>> convert_file_size_to_int("1MiB")
    1048576
    ```
    """
    if isinstance(size, int):
        return size
    if size.upper().endswith('GIB'):
        return int(size[:-3]) * 2 ** 30
    if size.upper().endswith('MIB'):
        return int(size[:-3]) * 2 ** 20
    if size.upper().endswith('KIB'):
        return int(size[:-3]) * 2 ** 10
    if size.upper().endswith('GB'):
        int_size = int(size[:-2]) * 10 ** 9
        return int_size // 8 if size.endswith('b') else int_size
    if size.upper().endswith('MB'):
        int_size = int(size[:-2]) * 10 ** 6
        return int_size // 8 if size.endswith('b') else int_size
    if size.upper().endswith('KB'):
        int_size = int(size[:-2]) * 10 ** 3
        return int_size // 8 if size.endswith('b') else int_size
    raise ValueError("`size` is not in a valid format. Use an integer followed by the unit, e.g., '5GB'.")