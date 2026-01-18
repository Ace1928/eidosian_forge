from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name, name_sequence
def remove_within_group_for_percentiles(expression: exp.Expression) -> exp.Expression:
    """Transforms percentiles by getting rid of their corresponding WITHIN GROUP clause."""
    if isinstance(expression, exp.WithinGroup) and isinstance(expression.this, PERCENTILES) and isinstance(expression.expression, exp.Order):
        quantile = expression.this.this
        input_value = t.cast(exp.Ordered, expression.find(exp.Ordered)).this
        return expression.replace(exp.ApproxQuantile(this=input_value, quantile=quantile))
    return expression