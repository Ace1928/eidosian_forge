import collections.abc
import gc
import inspect
import re
import sys
import weakref
from functools import partial, wraps
from itertools import chain
from typing import (
from scrapy.utils.asyncgen import as_async_generator
def binary_is_text(data: bytes) -> bool:
    """Returns ``True`` if the given ``data`` argument (a ``bytes`` object)
    does not contain unprintable control characters.
    """
    if not isinstance(data, bytes):
        raise TypeError(f"data must be bytes, got '{type(data).__name__}'")
    return all((c not in _BINARYCHARS for c in data))