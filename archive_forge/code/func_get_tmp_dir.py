import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
from .folder import ImageFolder
from .utils import check_integrity, extract_archive, verify_str_arg
@contextmanager
def get_tmp_dir() -> Iterator[str]:
    tmp_dir = tempfile.mkdtemp()
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)