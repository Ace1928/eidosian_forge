from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
@_add_atomic_docstr('compare-and-swap', has_cmp=True)
def atomic_cas(pointer, cmp, val, sem=None, scope=None, _builder=None):
    cmp = _to_tensor(cmp, _builder)
    val = _to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    scope = _constexpr_to_value(scope)
    return semantic.atomic_cas(pointer, cmp, val, sem, scope, _builder)