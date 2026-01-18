from __future__ import annotations
import datetime
import functools
import itertools
import typing as t
from collections import deque
from decimal import Decimal
from functools import reduce
import sqlglot
from sqlglot import Dialect, exp
from sqlglot.helper import first, merge_ranges, while_changing
from sqlglot.optimizer.scope import find_all_in_scope, walk_in_scope
def simplify_not(expression):
    """
    Demorgan's Law
    NOT (x OR y) -> NOT x AND NOT y
    NOT (x AND y) -> NOT x OR NOT y
    """
    if isinstance(expression, exp.Not):
        this = expression.this
        if is_null(this):
            return exp.null()
        if this.__class__ in COMPLEMENT_COMPARISONS:
            return COMPLEMENT_COMPARISONS[this.__class__](this=this.this, expression=this.expression)
        if isinstance(this, exp.Paren):
            condition = this.unnest()
            if isinstance(condition, exp.And):
                return exp.paren(exp.or_(exp.not_(condition.left, copy=False), exp.not_(condition.right, copy=False), copy=False))
            if isinstance(condition, exp.Or):
                return exp.paren(exp.and_(exp.not_(condition.left, copy=False), exp.not_(condition.right, copy=False), copy=False))
            if is_null(condition):
                return exp.null()
        if always_true(this):
            return exp.false()
        if is_false(this):
            return exp.true()
        if isinstance(this, exp.Not):
            return this.this
    return expression