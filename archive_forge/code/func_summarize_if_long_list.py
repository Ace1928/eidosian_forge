import os
import types
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pyarrow as pa
from filelock import BaseFileLock, Timeout
from . import config
from .arrow_dataset import Dataset
from .arrow_reader import ArrowReader
from .arrow_writer import ArrowWriter
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadManager
from .features import Features
from .info import DatasetInfo, MetricInfo
from .naming import camelcase_to_snakecase
from .utils._filelock import FileLock
from .utils.deprecation_utils import deprecated
from .utils.logging import get_logger
from .utils.py_utils import copyfunc, temp_seed
def summarize_if_long_list(obj):
    if not type(obj) == list or len(obj) <= 6:
        return f'{obj}'

    def format_chunk(chunk):
        return ', '.join((repr(x) for x in chunk))
    return f'[{format_chunk(obj[:3])}, ..., {format_chunk(obj[-3:])}]'