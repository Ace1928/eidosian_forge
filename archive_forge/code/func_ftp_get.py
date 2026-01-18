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
def ftp_get(url, temp_file, timeout=10.0):
    _raise_if_offline_mode_is_enabled(f'Tried to reach {url}')
    try:
        logger.info(f'Getting through FTP {url} into {temp_file.name}')
        with closing(urllib.request.urlopen(url, timeout=timeout)) as r:
            shutil.copyfileobj(r, temp_file)
    except urllib.error.URLError as e:
        raise ConnectionError(e) from None