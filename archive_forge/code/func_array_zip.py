from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def array_zip(*cols: ColumnOrName) -> Column:
    if len(cols) == 1:
        return Column.invoke_anonymous_function(cols[0], 'ARRAY_ZIP')
    return Column.invoke_anonymous_function(cols[0], 'ARRAY_ZIP', *cols[1:])