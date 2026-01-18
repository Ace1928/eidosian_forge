from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def covar_samp(col1: ColumnOrName, col2: ColumnOrName) -> Column:
    return Column.invoke_expression_over_column(col1, expression.CovarSamp, expression=col2)