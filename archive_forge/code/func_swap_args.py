from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def swap_args(func: BinaryCoercionFunc) -> BinaryCoercionFunc:

    @functools.wraps(func)
    def _swapped(l: exp.Expression, r: exp.Expression) -> exp.DataType.Type:
        return func(r, l)
    return _swapped