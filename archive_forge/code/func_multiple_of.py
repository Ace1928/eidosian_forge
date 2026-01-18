from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
def multiple_of(input, values, _builder=None):
    """
    Let the compiler know that the values in :code:`input` are all multiples of :code:`value`.
    """
    if isinstance(values, constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, constexpr):
            raise TypeError(f'values element {i} must have type `constexpr`')
        if not isinstance(d.value, int):
            raise TypeError(f'values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]')
    values = [x.value for x in values]
    return semantic.multiple_of(input, values)