from __future__ import annotations
import abc
import tempfile
from pathlib import Path, PurePath
from lazyops.utils.pooler import ThreadPoolV2
from lazyops.utils.logs import default_logger, null_logger, Logger
from typing import Any, Union, Optional, Type, Iterable, Callable, Dict, List, Tuple, TypeVar, TYPE_CHECKING
@classmethod
def convert_text_to_bytes(cls, data: str, encoding: Optional[str]='utf-8', errors: Optional[str]='ignore', **kwargs) -> bytes:
    """
        Converts file text to bytes
        """
    return data.encode(encoding, errors=errors)