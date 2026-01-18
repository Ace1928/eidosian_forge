from __future__ import annotations
import typing as t
import sqlglot
from sqlglot import expressions as exp
from sqlglot.dataframe.sql.types import DataType
from sqlglot.helper import flatten, is_iterable
def asc_nulls_last(self) -> Column:
    new_expression = exp.Ordered(this=self.column_expression, desc=False, nulls_first=False)
    return Column(new_expression)