from __future__ import annotations
import datetime
import inspect
import logging
import re
import sys
import typing as t
from collections.abc import Collection, Set
from contextlib import contextmanager
from copy import copy
from enum import Enum
from itertools import count
def apply_index_offset(this: exp.Expression, expressions: t.List[E], offset: int) -> t.List[E]:
    """
    Applies an offset to a given integer literal expression.

    Args:
        this: The target of the index.
        expressions: The expression the offset will be applied to, wrapped in a list.
        offset: The offset that will be applied.

    Returns:
        The original expression with the offset applied to it, wrapped in a list. If the provided
        `expressions` argument contains more than one expression, it's returned unaffected.
    """
    if not offset or len(expressions) != 1:
        return expressions
    expression = expressions[0]
    from sqlglot import exp
    from sqlglot.optimizer.annotate_types import annotate_types
    from sqlglot.optimizer.simplify import simplify
    if not this.type:
        annotate_types(this)
    if t.cast(exp.DataType, this.type).this not in (exp.DataType.Type.UNKNOWN, exp.DataType.Type.ARRAY):
        return expressions
    if not expression.type:
        annotate_types(expression)
    if t.cast(exp.DataType, expression.type).this in exp.DataType.INTEGER_TYPES:
        logger.warning('Applying array index offset (%s)', offset)
        expression = simplify(expression + offset)
        return [expression]
    return expressions