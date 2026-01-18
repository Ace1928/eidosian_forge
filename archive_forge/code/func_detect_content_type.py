from __future__ import annotations
import abc
import tempfile
from pathlib import Path, PurePath
from lazyops.utils.pooler import ThreadPoolV2
from lazyops.utils.logs import default_logger, null_logger, Logger
from typing import Any, Union, Optional, Type, Iterable, Callable, Dict, List, Tuple, TypeVar, TYPE_CHECKING
@classmethod
def detect_content_type(cls, data: InputContentType, mime: Optional[bool]=False, **kwargs) -> str:
    """
        Detect the content type of the data
        """
    from lazyops.imports._filemagic import resolve_magic
    resolve_magic(True)
    from magic import from_file, from_buffer
    if isinstance(data, str):
        return from_file(data, mime=mime)
    if isinstance(data, bytes):
        return from_buffer(data, mime=mime)
    return from_buffer(cls.convert_file_input_to_bytes(data), mime=mime)