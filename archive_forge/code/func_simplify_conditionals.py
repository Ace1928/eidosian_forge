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
def simplify_conditionals(expression):
    """Simplifies expressions like IF, CASE if their condition is statically known."""
    if isinstance(expression, exp.Case):
        this = expression.this
        for case in expression.args['ifs']:
            cond = case.this
            if this:
                cond = cond.replace(this.pop().eq(cond))
            if always_true(cond):
                return case.args['true']
            if always_false(cond):
                case.pop()
                if not expression.args['ifs']:
                    return expression.args.get('default') or exp.null()
    elif isinstance(expression, exp.If) and (not isinstance(expression.parent, exp.Case)):
        if always_true(expression.this):
            return expression.args['true']
        if always_false(expression.this):
            return expression.args.get('false') or exp.null()
    return expression