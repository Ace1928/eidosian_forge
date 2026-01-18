from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
def binary_op_type_legalization(lhs, rhs, builder):
    """
        Convert both operands to a single common type
        :param lhs: the left operand
        :param rhs: the right operand
        :param builder: the builder
    """
    return semantic.binary_op_type_checking_impl(lhs, rhs, builder)