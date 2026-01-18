from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name, name_sequence
def epoch_cast_to_ts(expression: exp.Expression) -> exp.Expression:
    """Replace 'epoch' in casts by the equivalent date literal."""
    if isinstance(expression, (exp.Cast, exp.TryCast)) and expression.name.lower() == 'epoch' and (expression.to.this in exp.DataType.TEMPORAL_TYPES):
        expression.this.replace(exp.Literal.string('1970-01-01 00:00:00'))
    return expression