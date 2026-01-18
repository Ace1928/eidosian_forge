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
def http_get(url, temp_file, proxies=None, resume_size=0, headers=None, cookies=None, timeout=100.0, max_retries=0, desc=None) -> Optional[requests.Response]:
    headers = dict(headers) if headers is not None else {}
    headers['user-agent'] = get_datasets_user_agent(user_agent=headers.get('user-agent'))
    if resume_size > 0:
        headers['Range'] = f'bytes={resume_size:d}-'
    response = _request_with_retry(method='GET', url=url, stream=True, proxies=proxies, headers=headers, cookies=cookies, max_retries=max_retries, timeout=timeout)
    if temp_file is None:
        return response
    if response.status_code == 416:
        return
    content_length = response.headers.get('Content-Length')
    total = resume_size + int(content_length) if content_length is not None else None
    with hf_tqdm(unit='B', unit_scale=True, total=total, initial=resume_size, desc=desc or 'Downloading', position=multiprocessing.current_process()._identity[-1] if os.environ.get('HF_DATASETS_STACK_MULTIPROCESSING_DOWNLOAD_PROGRESS_BARS') == '1' and multiprocessing.current_process()._identity else None) as progress:
        for chunk in response.iter_content(chunk_size=1024):
            progress.update(len(chunk))
            temp_file.write(chunk)