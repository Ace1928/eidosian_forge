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
@catch(ModuleNotFoundError, UnsupportedUnit)
def simplify_datetrunc(expression: exp.Expression, dialect: Dialect) -> exp.Expression:
    """Simplify expressions like `DATE_TRUNC('year', x) >= CAST('2021-01-01' AS DATE)`"""
    comparison = expression.__class__
    if isinstance(expression, DATETRUNCS):
        this = expression.this
        trunc_type = extract_type(this)
        date = extract_date(this)
        if date and expression.unit:
            return date_literal(date_floor(date, expression.unit.name.lower(), dialect), trunc_type)
    elif comparison not in DATETRUNC_COMPARISONS:
        return expression
    if isinstance(expression, exp.Binary):
        l, r = (expression.left, expression.right)
        if not _is_datetrunc_predicate(l, r):
            return expression
        l = t.cast(exp.DateTrunc, l)
        trunc_arg = l.this
        unit = l.unit.name.lower()
        date = extract_date(r)
        if not date:
            return expression
        return DATETRUNC_BINARY_COMPARISONS[comparison](trunc_arg, date, unit, dialect, extract_type(trunc_arg, r)) or expression
    if isinstance(expression, exp.In):
        l = expression.this
        rs = expression.expressions
        if rs and all((_is_datetrunc_predicate(l, r) for r in rs)):
            l = t.cast(exp.DateTrunc, l)
            unit = l.unit.name.lower()
            ranges = []
            for r in rs:
                date = extract_date(r)
                if not date:
                    return expression
                drange = _datetrunc_range(date, unit, dialect)
                if drange:
                    ranges.append(drange)
            if not ranges:
                return expression
            ranges = merge_ranges(ranges)
            target_type = extract_type(l, *rs)
            return exp.or_(*[_datetrunc_eq_expression(l, drange, target_type) for drange in ranges], copy=False)
    return expression