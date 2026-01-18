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
def init_hf_modules(hf_modules_cache: Optional[Union[Path, str]]=None) -> str:
    """
    Add hf_modules_cache to the python path.
    By default hf_modules_cache='~/.cache/huggingface/modules'.
    It can also be set with the environment variable HF_MODULES_CACHE.
    This is used to add modules such as `datasets_modules`
    """
    hf_modules_cache = hf_modules_cache if hf_modules_cache is not None else config.HF_MODULES_CACHE
    hf_modules_cache = str(hf_modules_cache)
    if hf_modules_cache not in sys.path:
        sys.path.append(hf_modules_cache)
        os.makedirs(hf_modules_cache, exist_ok=True)
        if not os.path.exists(os.path.join(hf_modules_cache, '__init__.py')):
            with open(os.path.join(hf_modules_cache, '__init__.py'), 'w'):
                pass
    return hf_modules_cache