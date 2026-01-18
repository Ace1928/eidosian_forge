from __future__ import annotations
import abc
import tempfile
from pathlib import Path, PurePath
from lazyops.utils.pooler import ThreadPoolV2
from lazyops.utils.logs import default_logger, null_logger, Logger
from typing import Any, Union, Optional, Type, Iterable, Callable, Dict, List, Tuple, TypeVar, TYPE_CHECKING
@classmethod
def convert_bytes_to_file(cls, data: bytes, path: Optional[InputPathType]=None, make_temp: Optional[bool]=False, **kwargs) -> 'FileLike':
    """
        Converts file bytes to a file
        """
    path = File(path) if path and (not make_temp) else File(tempfile.mktemp())
    path.write_bytes(data)
    return path