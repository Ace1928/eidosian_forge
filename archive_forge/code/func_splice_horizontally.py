from __future__ import annotations
import collections
import functools
import operator
import typing
from typing import Any
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .result import IteratorResult
from .result import MergedResult
from .result import Result
from .result import ResultMetaData
from .result import SimpleResultMetaData
from .result import tuplegetter
from .row import Row
from .. import exc
from .. import util
from ..sql import elements
from ..sql import sqltypes
from ..sql import util as sql_util
from ..sql.base import _generative
from ..sql.compiler import ResultColumnsEntry
from ..sql.compiler import RM_NAME
from ..sql.compiler import RM_OBJECTS
from ..sql.compiler import RM_RENDERED_NAME
from ..sql.compiler import RM_TYPE
from ..sql.type_api import TypeEngine
from ..util import compat
from ..util.typing import Literal
from ..util.typing import Self
def splice_horizontally(self, other):
    """Return a new :class:`.CursorResult` that "horizontally splices"
        together the rows of this :class:`.CursorResult` with that of another
        :class:`.CursorResult`.

        .. tip::  This method is for the benefit of the SQLAlchemy ORM and is
           not intended for general use.

        "horizontally splices" means that for each row in the first and second
        result sets, a new row that concatenates the two rows together is
        produced, which then becomes the new row.  The incoming
        :class:`.CursorResult` must have the identical number of rows.  It is
        typically expected that the two result sets come from the same sort
        order as well, as the result rows are spliced together based on their
        position in the result.

        The expected use case here is so that multiple INSERT..RETURNING
        statements (which definitely need to be sorted) against different
        tables can produce a single result that looks like a JOIN of those two
        tables.

        E.g.::

            r1 = connection.execute(
                users.insert().returning(
                    users.c.user_name,
                    users.c.user_id,
                    sort_by_parameter_order=True
                ),
                user_values
            )

            r2 = connection.execute(
                addresses.insert().returning(
                    addresses.c.address_id,
                    addresses.c.address,
                    addresses.c.user_id,
                    sort_by_parameter_order=True
                ),
                address_values
            )

            rows = r1.splice_horizontally(r2).all()
            assert (
                rows ==
                [
                    ("john", 1, 1, "foo@bar.com", 1),
                    ("jack", 2, 2, "bar@bat.com", 2),
                ]
            )

        .. versionadded:: 2.0

        .. seealso::

            :meth:`.CursorResult.splice_vertically`


        """
    clone = self._generate()
    total_rows = [tuple(r1) + tuple(r2) for r1, r2 in zip(list(self._raw_row_iterator()), list(other._raw_row_iterator()))]
    clone._metadata = clone._metadata._splice_horizontally(other._metadata)
    clone.cursor_strategy = FullyBufferedCursorFetchStrategy(None, initial_buffer=total_rows)
    clone._reset_memoizations()
    return clone