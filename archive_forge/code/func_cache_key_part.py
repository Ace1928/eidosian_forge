from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@property
def cache_key_part(self) -> str:
    """See cache_key_part() in triton.cc."""
    return self.name