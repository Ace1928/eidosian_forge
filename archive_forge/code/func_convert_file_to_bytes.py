from __future__ import annotations
import abc
import tempfile
from pathlib import Path, PurePath
from lazyops.utils.pooler import ThreadPoolV2
from lazyops.utils.logs import default_logger, null_logger, Logger
from typing import Any, Union, Optional, Type, Iterable, Callable, Dict, List, Tuple, TypeVar, TYPE_CHECKING
@classmethod
def convert_file_to_bytes(cls, path: InputPathType, **kwargs) -> bytes:
    """
        Converts a file to bytes
        """
    return Path(path).read_bytes()