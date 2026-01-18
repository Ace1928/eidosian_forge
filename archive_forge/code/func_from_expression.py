from __future__ import annotations
import math
import typing as t
from sqlglot import alias, exp
from sqlglot.helper import name_sequence
from sqlglot.optimizer.eliminate_joins import join_condition
@classmethod
def from_expression(cls, expression: exp.Expression, ctes: t.Optional[t.Dict[str, Step]]=None) -> SetOperation:
    assert isinstance(expression, exp.Union)
    left = Step.from_expression(expression.left, ctes)
    left.name = left.name or 'left'
    right = Step.from_expression(expression.right, ctes)
    right.name = right.name or 'right'
    step = cls(op=expression.__class__, left=left.name, right=right.name, distinct=bool(expression.args.get('distinct')))
    step.add_dependency(left)
    step.add_dependency(right)
    limit = expression.args.get('limit')
    if limit:
        step.limit = int(limit.text('expression'))
    return step