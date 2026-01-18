from __future__ import annotations
import typing as t
import sqlglot
from sqlglot import expressions as exp
from sqlglot.dataframe.sql.types import DataType
from sqlglot.helper import flatten, is_iterable
@classmethod
def invoke_expression_over_column(cls, column: t.Optional[ColumnOrLiteral], callable_expression: t.Callable, **kwargs) -> Column:
    ensured_column = None if column is None else cls.ensure_col(column)
    ensure_expression_values = {k: [Column.ensure_col(x).expression for x in v] if is_iterable(v) else Column.ensure_col(v).expression for k, v in kwargs.items() if v is not None}
    new_expression = callable_expression(**ensure_expression_values) if ensured_column is None else callable_expression(this=ensured_column.column_expression, **ensure_expression_values)
    return Column(new_expression)