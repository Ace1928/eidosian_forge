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
def hash_url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name
    so that TF 2.0 can identify it as a HDF5 file
    (see https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    url_bytes = url.encode('utf-8')
    url_hash = insecure_hashlib.sha256(url_bytes)
    filename = url_hash.hexdigest()
    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = insecure_hashlib.sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()
    if url.endswith('.py'):
        filename += '.py'
    return filename