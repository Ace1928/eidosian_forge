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
@property
def inserted_primary_key(self):
    """Return the primary key for the row just inserted.

        The return value is a :class:`_result.Row` object representing
        a named tuple of primary key values in the order in which the
        primary key columns are configured in the source
        :class:`_schema.Table`.

        .. versionchanged:: 1.4.8 - the
           :attr:`_engine.CursorResult.inserted_primary_key`
           value is now a named tuple via the :class:`_result.Row` class,
           rather than a plain tuple.

        This accessor only applies to single row :func:`_expression.insert`
        constructs which did not explicitly specify
        :meth:`_expression.Insert.returning`.    Support for multirow inserts,
        while not yet available for most backends, would be accessed using
        the :attr:`_engine.CursorResult.inserted_primary_key_rows` accessor.

        Note that primary key columns which specify a server_default clause, or
        otherwise do not qualify as "autoincrement" columns (see the notes at
        :class:`_schema.Column`), and were generated using the database-side
        default, will appear in this list as ``None`` unless the backend
        supports "returning" and the insert statement executed with the
        "implicit returning" enabled.

        Raises :class:`~sqlalchemy.exc.InvalidRequestError` if the executed
        statement is not a compiled expression construct
        or is not an insert() construct.

        """
    if self.context.executemany:
        raise exc.InvalidRequestError('This statement was an executemany call; if primary key returning is supported, please use .inserted_primary_key_rows.')
    ikp = self.inserted_primary_key_rows
    if ikp:
        return ikp[0]
    else:
        return None