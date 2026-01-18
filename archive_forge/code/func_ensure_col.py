from __future__ import annotations
import typing as t
import sqlglot
from sqlglot import expressions as exp
from sqlglot.dataframe.sql.types import DataType
from sqlglot.helper import flatten, is_iterable
@classmethod
def ensure_col(cls, value: t.Optional[t.Union[ColumnOrLiteral, exp.Expression]]) -> Column:
    return cls(value)