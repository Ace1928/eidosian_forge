from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
def static_print(*values, sep: str=' ', end: str='\n', file=None, flush=False, _builder=None):
    """
    Print the values at compile time.  The parameters are the same as the builtin :code:`print`.

    NOTE: Calling the Python builtin :code:`print` is not the same as calling this, it instead maps to :code:`device_print`,
    which has special requirements for the arguments.

    .. highlight:: python
    .. code-block:: python

        tl.static_print(f"{BLOCK_SIZE=}")
    """
    pass