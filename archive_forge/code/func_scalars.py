from __future__ import annotations
import operator
from typing import Any
from typing import AsyncIterator
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from . import exc as async_exc
from ... import util
from ...engine import Result
from ...engine.result import _NO_ROW
from ...engine.result import _R
from ...engine.result import _WithKeys
from ...engine.result import FilterResult
from ...engine.result import FrozenResult
from ...engine.result import ResultMetaData
from ...engine.row import Row
from ...engine.row import RowMapping
from ...sql.base import _generative
from ...util.concurrency import greenlet_spawn
from ...util.typing import Literal
from ...util.typing import Self
def scalars(self, index: _KeyIndexType=0) -> AsyncScalarResult[Any]:
    """Return an :class:`_asyncio.AsyncScalarResult` filtering object which
        will return single elements rather than :class:`_row.Row` objects.

        Refer to :meth:`_result.Result.scalars` in the synchronous
        SQLAlchemy API for a complete behavioral description.

        :param index: integer or row key indicating the column to be fetched
         from each row, defaults to ``0`` indicating the first column.

        :return: a new :class:`_asyncio.AsyncScalarResult` filtering object
         referring to this :class:`_asyncio.AsyncResult` object.

        """
    return AsyncScalarResult(self._real_result, index)