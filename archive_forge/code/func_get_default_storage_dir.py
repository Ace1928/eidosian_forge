import atexit
from hashlib import md5
import mimetypes
import os
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryDirectory
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from ..client import Client
from ..enums import FileCacheMode
from .localpath import LocalPath
@classmethod
def get_default_storage_dir(cls) -> Path:
    if cls._default_storage_temp_dir is None:
        cls._default_storage_temp_dir = TemporaryDirectory()
        _temp_dirs_to_clean.append(cls._default_storage_temp_dir)
    return Path(cls._default_storage_temp_dir.name)