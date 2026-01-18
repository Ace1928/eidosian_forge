from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name, name_sequence
def eliminate_semi_and_anti_joins(expression: exp.Expression) -> exp.Expression:
    """Convert SEMI and ANTI joins into equivalent forms that use EXIST instead."""
    if isinstance(expression, exp.Select):
        for join in expression.args.get('joins') or []:
            on = join.args.get('on')
            if on and join.kind in ('SEMI', 'ANTI'):
                subquery = exp.select('1').from_(join.this).where(on)
                exists = exp.Exists(this=subquery)
                if join.kind == 'ANTI':
                    exists = exists.not_(copy=False)
                join.pop()
                expression.where(exists, copy=False)
    return expression