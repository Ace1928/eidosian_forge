import copy
import io
import json
import multiprocessing
import os
import posixpath
import re
import shutil
import sys
import time
import urllib
import warnings
from contextlib import closing, contextmanager
from functools import partial
from pathlib import Path
from typing import Optional, TypeVar, Union
from unittest.mock import patch
from urllib.parse import urljoin, urlparse
import fsspec
import huggingface_hub
import requests
from fsspec.core import strip_protocol
from fsspec.utils import can_be_local
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from .. import __version__, config
from ..download.download_config import DownloadConfig
from . import _tqdm, logging
from . import tqdm as hf_tqdm
from ._filelock import FileLock
from .extract import ExtractManager
def fsspec_get(url, temp_file, storage_options=None, desc=None):
    _raise_if_offline_mode_is_enabled(f'Tried to reach {url}')
    fs, _, paths = fsspec.get_fs_token_paths(url, storage_options=storage_options)
    if len(paths) > 1:
        raise ValueError(f'GET can be called with at most one path but was called with {paths}')
    callback = TqdmCallback(tqdm_kwargs={'desc': desc or 'Downloading', 'unit': 'B', 'unit_scale': True, 'position': multiprocessing.current_process()._identity[-1] if os.environ.get('HF_DATASETS_STACK_MULTIPROCESSING_DOWNLOAD_PROGRESS_BARS') == '1' and multiprocessing.current_process()._identity else None})
    fs.get_file(paths[0], temp_file.name, callback=callback)