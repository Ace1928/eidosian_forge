from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
def fdiv(x, y, ieee_rounding=False, _builder=None):
    """
    Returns a floating-point resultant tensor of dividing x by y.

    :param x: the input numerator value.
    :param y: the input denominator value.
    :param ieee_rounding: To follow IEEE-754 floating point number
        rounding mechanism
    :type ieee_rounding: bool
    """
    ieee_rounding = _constexpr_to_value(ieee_rounding)
    x = _to_tensor(x, _builder)
    y = _to_tensor(y, _builder)
    return semantic.fdiv(x, y, ieee_rounding, _builder)