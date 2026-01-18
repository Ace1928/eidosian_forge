from __future__ import annotations
import re
import sys
import warnings
from functools import wraps, lru_cache
from itertools import count
from typing import TYPE_CHECKING, Generic, Iterator, NamedTuple, TypeVar, TypedDict, overload
def nearing_recursion_limit() -> bool:
    """Return true if current stack depth is within 100 of maximum limit."""
    return sys.getrecursionlimit() - _get_stack_depth() < 100