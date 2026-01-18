import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
from .folder import ImageFolder
from .utils import check_integrity, extract_archive, verify_str_arg
@property
def split_folder(self) -> str:
    return os.path.join(self.root, self.split)