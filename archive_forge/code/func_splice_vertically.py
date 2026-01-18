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
def splice_vertically(self, other):
    """Return a new :class:`.CursorResult` that "vertically splices",
        i.e. "extends", the rows of this :class:`.CursorResult` with that of
        another :class:`.CursorResult`.

        .. tip::  This method is for the benefit of the SQLAlchemy ORM and is
           not intended for general use.

        "vertically splices" means the rows of the given result are appended to
        the rows of this cursor result. The incoming :class:`.CursorResult`
        must have rows that represent the identical list of columns in the
        identical order as they are in this :class:`.CursorResult`.

        .. versionadded:: 2.0

        .. seealso::

            :meth:`.CursorResult.splice_horizontally`

        """
    clone = self._generate()
    total_rows = list(self._raw_row_iterator()) + list(other._raw_row_iterator())
    clone.cursor_strategy = FullyBufferedCursorFetchStrategy(None, initial_buffer=total_rows)
    clone._reset_memoizations()
    return clone