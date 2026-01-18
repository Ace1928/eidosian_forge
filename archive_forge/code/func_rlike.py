from __future__ import annotations
import typing as t
import sqlglot
from sqlglot import expressions as exp
from sqlglot.dataframe.sql.types import DataType
from sqlglot.helper import flatten, is_iterable
def rlike(self, regexp: str) -> Column:
    return self.invoke_expression_over_column(column=self, callable_expression=exp.RegexpLike, expression=self._lit(regexp).expression)