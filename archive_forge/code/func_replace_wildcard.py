from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression
from triad.utils.schema import quote_name
from fugue.column.expressions import (
from fugue.column.functions import is_agg
from fugue.exceptions import FugueBug
def replace_wildcard(self, schema: Schema) -> 'SelectColumns':
    """Replace wildcard ``*`` with explicit column names

        :param schema: the schema used to parse the wildcard
        :return: a new instance containing only explicit columns

        .. note::

            It only replaces the top level ``*``. For example
            ``count_distinct(all_cols())`` will not be transformed because
            this ``*`` is not first level.
        """

    def _get_cols() -> Iterable[ColumnExpr]:
        for c in self.all_cols:
            if isinstance(c, _WildcardExpr):
                yield from [col(n) for n in schema.names]
            else:
                yield c
    return SelectColumns(*list(_get_cols()))