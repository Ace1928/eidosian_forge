from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def overlay(src: ColumnOrName, replace: ColumnOrName, pos: t.Union[ColumnOrName, int], len: t.Optional[t.Union[ColumnOrName, int]]=None) -> Column:
    if len is not None:
        return Column.invoke_anonymous_function(src, 'OVERLAY', replace, pos, len)
    return Column.invoke_anonymous_function(src, 'OVERLAY', replace, pos)