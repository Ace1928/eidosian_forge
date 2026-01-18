from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def session_window(timeColumn: ColumnOrName, gapDuration: ColumnOrName) -> Column:
    gap_duration_column = gapDuration if isinstance(gapDuration, Column) else lit(gapDuration)
    return Column.invoke_anonymous_function(timeColumn, 'SESSION_WINDOW', gap_duration_column)