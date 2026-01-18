from __future__ import annotations
import abc
import tempfile
from pathlib import Path, PurePath
from lazyops.utils.pooler import ThreadPoolV2
from lazyops.utils.logs import default_logger, null_logger, Logger
from typing import Any, Union, Optional, Type, Iterable, Callable, Dict, List, Tuple, TypeVar, TYPE_CHECKING
@classmethod
def convert_file_input_to_bytes(cls, data: InputContentType, **kwargs) -> bytes:
    """
        Converts input to bytes
        """
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return Path(data).read_bytes()
    if isinstance(data, Iterable):
        return b''.join(data)
    if isinstance(data, Path) or hasattr(data, 'read_bytes'):
        return data.read_bytes()
    raise TypeError(f'Invalid data type: {type(data)}')