import bz2
import contextlib
import gzip
import hashlib
import itertools
import lzma
import os
import os.path
import pathlib
import re
import sys
import tarfile
import urllib
import urllib.error
import urllib.request
import warnings
import zipfile
from typing import Any, Callable, Dict, IO, Iterable, Iterator, List, Optional, Tuple, TypeVar
from urllib.parse import urlparse
import numpy as np
import requests
import torch
from torch.utils.model_zoo import tqdm
from .._internally_replaced_utils import _download_file_from_remote_location, _is_remote_location_available
def calculate_md5(fpath: str, chunk_size: int=1024 * 1024) -> str:
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        while (chunk := f.read(chunk_size)):
            md5.update(chunk)
    return md5.hexdigest()