from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.dataframe.sql import types
def get_tables_from_expression_with_join(expression: exp.Select) -> t.List[exp.Table]:
    if not expression.args.get('joins'):
        return []
    left_table = expression.args['from'].this
    other_tables = [join.this for join in expression.args['joins']]
    return [left_table] + other_tables