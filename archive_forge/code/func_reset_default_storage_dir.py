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
def reset_default_storage_dir(cls) -> Path:
    cls._default_storage_temp_dir = None
    return cls.get_default_storage_dir()