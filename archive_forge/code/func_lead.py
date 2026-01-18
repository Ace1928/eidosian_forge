from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def lead(col: ColumnOrName, offset: t.Optional[int]=1, default: t.Optional[t.Any]=None) -> Column:
    return Column.invoke_expression_over_column(col, expression.Lead, offset=None if offset == 1 else offset, default=default)