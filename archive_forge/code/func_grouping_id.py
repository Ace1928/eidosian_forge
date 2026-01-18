from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def grouping_id(*cols: ColumnOrName) -> Column:
    if not cols:
        return Column.invoke_anonymous_function(None, 'GROUPING_ID')
    if len(cols) == 1:
        return Column.invoke_anonymous_function(cols[0], 'GROUPING_ID')
    return Column.invoke_anonymous_function(cols[0], 'GROUPING_ID', *cols[1:])