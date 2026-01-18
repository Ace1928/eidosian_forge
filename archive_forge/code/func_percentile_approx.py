from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def percentile_approx(col: ColumnOrName, percentage: t.Union[ColumnOrLiteral, t.List[float], t.Tuple[float]], accuracy: t.Optional[t.Union[ColumnOrLiteral, int]]=None) -> Column:
    if accuracy:
        return Column.invoke_expression_over_column(col, expression.ApproxQuantile, quantile=lit(percentage), accuracy=accuracy)
    return Column.invoke_expression_over_column(col, expression.ApproxQuantile, quantile=lit(percentage))