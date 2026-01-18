from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def regexp_extract(str: ColumnOrName, pattern: str, idx: t.Optional[int]=None) -> Column:
    return Column.invoke_expression_over_column(str, expression.RegexpExtract, expression=lit(pattern), group=idx)