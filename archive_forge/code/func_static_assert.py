from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
def static_assert(cond, msg='', _builder=None):
    """
    Assert the condition at compile time.  Does not require that the :code:`TRITON_DEBUG` environment variable
    is set.

    .. highlight:: python
    .. code-block:: python

        tl.static_assert(BLOCK_SIZE == 1024)
    """
    pass