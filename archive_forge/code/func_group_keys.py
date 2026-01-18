from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression
from triad.utils.schema import quote_name
from fugue.column.expressions import (
from fugue.column.functions import is_agg
from fugue.exceptions import FugueBug
@property
def group_keys(self) -> List[ColumnExpr]:
    """Group keys inferred from the columns.

        .. note::

            * if there is no aggregation, the result will be empty
            * it is :meth:`~.simple_cols` plus :meth:`~.non_agg_funcs`
        """
    return self._group_keys