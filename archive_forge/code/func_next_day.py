from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def next_day(col: ColumnOrName, dayOfWeek: str) -> Column:
    return Column.invoke_anonymous_function(col, 'NEXT_DAY', lit(dayOfWeek))