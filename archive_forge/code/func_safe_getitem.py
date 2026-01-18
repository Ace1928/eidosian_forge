import ast
import sys
import builtins
from typing import Dict, Any, Optional
from . import line as line_properties
from .inspection import getattr_safe
def safe_getitem(obj, index):
    """Safely tries to access obj[index]"""
    if type(obj) in (list, tuple, dict, bytes, str):
        try:
            return obj[index]
        except (KeyError, IndexError):
            raise EvaluationError(f"can't lookup key {index!r} on {obj!r}")
    raise ValueError(f'unsafe to lookup on object of type {type(obj)}')