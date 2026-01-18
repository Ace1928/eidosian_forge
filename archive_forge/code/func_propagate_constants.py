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
def propagate_constants(expression, root=True):
    """
    Propagate constants for conjunctions in DNF:

    SELECT * FROM t WHERE a = b AND b = 5 becomes
    SELECT * FROM t WHERE a = 5 AND b = 5

    Reference: https://www.sqlite.org/optoverview.html
    """
    if isinstance(expression, exp.And) and (root or not expression.same_parent) and sqlglot.optimizer.normalize.normalized(expression, dnf=True):
        constant_mapping = {}
        for expr in walk_in_scope(expression, prune=lambda node: isinstance(node, exp.If)):
            if isinstance(expr, exp.EQ):
                l, r = (expr.left, expr.right)
                if isinstance(l, exp.Column) and isinstance(r, exp.Literal):
                    constant_mapping[l] = (id(l), r)
        if constant_mapping:
            for column in find_all_in_scope(expression, exp.Column):
                parent = column.parent
                column_id, constant = constant_mapping.get(column) or (None, None)
                if column_id is not None and id(column) != column_id and (not (isinstance(parent, exp.Is) and isinstance(parent.expression, exp.Null))):
                    column.replace(constant.copy())
    return expression