from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression
from triad.utils.schema import quote_name
from fugue.column.expressions import (
from fugue.column.functions import is_agg
from fugue.exceptions import FugueBug
def assert_no_wildcard(self) -> 'SelectColumns':
    """Assert there is no ``*`` on first level columns

        :raises AssertionError: if ``all_cols()`` exists
        :return: the instance itself
        """
    assert not self._has_wildcard
    return self