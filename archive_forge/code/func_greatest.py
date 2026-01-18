from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def greatest(*cols: ColumnOrName) -> Column:
    if len(cols) > 1:
        return Column.invoke_expression_over_column(cols[0], expression.Greatest, expressions=cols[1:])
    return Column.invoke_expression_over_column(cols[0], expression.Greatest)