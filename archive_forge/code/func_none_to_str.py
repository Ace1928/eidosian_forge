import ctypes
import json
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional, cast
from ._typing import _F
from .core import _LIB, _check_call, c_str, py_str
def none_to_str(value: Optional[str]) -> str:
    return '' if value is None else value