from __future__ import annotations
import typing as t
import sqlglot
from sqlglot import expressions as exp
from sqlglot.dataframe.sql.types import DataType
from sqlglot.helper import flatten, is_iterable
def inverse_binary_op(self, klass: t.Callable, other: ColumnOrLiteral, **kwargs) -> Column:
    return Column(klass(this=Column(other).column_expression, expression=self.column_expression, **kwargs))