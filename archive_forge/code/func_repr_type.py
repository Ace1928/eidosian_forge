from __future__ import annotations
import inspect
import re
import types
from typing import Any
def repr_type(obj: Any) -> str:
    """Return a string representation of a value and its type for readable

    error messages.
    """
    the_type = type(obj)
    return f'{obj!r} {the_type!r}'