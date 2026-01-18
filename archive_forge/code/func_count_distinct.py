from typing import Any, Optional
import pyarrow as pa
from fugue.column.expressions import (
from triad import Schema
def count_distinct(col: ColumnExpr) -> ColumnExpr:
    """SQL ``COUNT DISTINCT`` function (aggregation)

    :param col: the column to find distinct element count

    .. note::

        * this function cannot infer type from ``col`` type
        * this function can infer alias from ``col``'s inferred alias

    .. admonition:: New Since
        :class: hint

        **0.6.0**

    .. admonition:: Examples

        .. code-block:: python

            import fugue.column.functions as f

            f.count_distinct(all_cols())  # COUNT(DISTINCT *)
            f.count_distinct(col("a"))  # COUNT(DISTINCT a) AS a

            # you can specify explicitly
            # CAST(COUNT(DISTINCT a) AS double) AS a
            f.count_distinct(col("a")).cast(float)
    """
    assert isinstance(col, ColumnExpr)
    return _UnaryAggFuncExpr('COUNT', col, arg_distinct=True)