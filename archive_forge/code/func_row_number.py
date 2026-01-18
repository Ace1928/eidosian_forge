from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def row_number() -> Column:
    return Column(expression.Anonymous(this='ROW_NUMBER'))