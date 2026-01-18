from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def sha2(col: ColumnOrName, numBits: int) -> Column:
    column = col if isinstance(col, Column) else lit(col)
    return Column.invoke_expression_over_column(column, expression.SHA2, length=lit(numBits))