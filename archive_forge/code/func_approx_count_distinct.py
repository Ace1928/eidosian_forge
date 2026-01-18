from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def approx_count_distinct(col: ColumnOrName, rsd: t.Optional[float]=None) -> Column:
    if rsd is None:
        return Column.invoke_expression_over_column(col, expression.ApproxDistinct)
    return Column.invoke_expression_over_column(col, expression.ApproxDistinct, accuracy=rsd)