from __future__ import annotations
import itertools
import typing as t
from sqlglot import exp
from sqlglot.helper import is_date_unit, is_iso_date, is_iso_datetime
def ensure_bools(expression: exp.Expression, replace_func: t.Callable[[exp.Expression], None]) -> exp.Expression:
    if isinstance(expression, exp.Connector):
        replace_func(expression.left)
        replace_func(expression.right)
    elif isinstance(expression, exp.Not):
        replace_func(expression.this)
    elif isinstance(expression, exp.If) and (not (isinstance(expression.parent, exp.Case) and expression.parent.this)):
        replace_func(expression.this)
    elif isinstance(expression, (exp.Where, exp.Having)):
        replace_func(expression.this)
    return expression