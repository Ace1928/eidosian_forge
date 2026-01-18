from __future__ import annotations
import typing as t
import sqlglot
from sqlglot import expressions as exp
from sqlglot.dataframe.sql.types import DataType
from sqlglot.helper import flatten, is_iterable
@classmethod
def ensure_cols(cls, args: t.List[t.Union[ColumnOrLiteral, exp.Expression]]) -> t.List[Column]:
    return [cls.ensure_col(x) if not isinstance(x, Column) else x for x in args]