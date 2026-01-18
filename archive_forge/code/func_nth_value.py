from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def nth_value(col: ColumnOrName, offset: t.Optional[int]=1, ignoreNulls: t.Optional[bool]=None) -> Column:
    this = Column.invoke_expression_over_column(col, expression.NthValue, offset=None if offset == 1 else offset)
    if ignoreNulls is not None:
        return Column.invoke_expression_over_column(this, expression.IgnoreNulls)
    return this