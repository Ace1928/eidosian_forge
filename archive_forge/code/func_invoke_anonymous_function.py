from __future__ import annotations
import typing as t
import sqlglot
from sqlglot import expressions as exp
from sqlglot.dataframe.sql.types import DataType
from sqlglot.helper import flatten, is_iterable
@classmethod
def invoke_anonymous_function(cls, column: t.Optional[ColumnOrLiteral], func_name: str, *args: t.Optional[ColumnOrLiteral]) -> Column:
    columns = [] if column is None else [cls.ensure_col(column)]
    column_args = [cls.ensure_col(arg) for arg in args]
    expressions = [x.expression for x in columns + column_args]
    new_expression = exp.Anonymous(this=func_name.upper(), expressions=expressions)
    return Column(new_expression)