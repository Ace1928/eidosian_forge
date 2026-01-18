from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name, name_sequence
def struct_kv_to_alias(expression: exp.Expression) -> exp.Expression:
    """Converts struct arguments to aliases, e.g. STRUCT(1 AS y)."""
    if isinstance(expression, exp.Struct):
        expression.set('expressions', [exp.alias_(e.expression, e.this) if isinstance(e, exp.PropertyEQ) else e for e in expression.expressions])
    return expression