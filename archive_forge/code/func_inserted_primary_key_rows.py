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
def inserted_primary_key_rows(self):
    """Return the value of
        :attr:`_engine.CursorResult.inserted_primary_key`
        as a row contained within a list; some dialects may support a
        multiple row form as well.

        .. note:: As indicated below, in current SQLAlchemy versions this
           accessor is only useful beyond what's already supplied by
           :attr:`_engine.CursorResult.inserted_primary_key` when using the
           :ref:`postgresql_psycopg2` dialect.   Future versions hope to
           generalize this feature to more dialects.

        This accessor is added to support dialects that offer the feature
        that is currently implemented by the :ref:`psycopg2_executemany_mode`
        feature, currently **only the psycopg2 dialect**, which provides
        for many rows to be INSERTed at once while still retaining the
        behavior of being able to return server-generated primary key values.

        * **When using the psycopg2 dialect, or other dialects that may support
          "fast executemany" style inserts in upcoming releases** : When
          invoking an INSERT statement while passing a list of rows as the
          second argument to :meth:`_engine.Connection.execute`, this accessor
          will then provide a list of rows, where each row contains the primary
          key value for each row that was INSERTed.

        * **When using all other dialects / backends that don't yet support
          this feature**: This accessor is only useful for **single row INSERT
          statements**, and returns the same information as that of the
          :attr:`_engine.CursorResult.inserted_primary_key` within a
          single-element list. When an INSERT statement is executed in
          conjunction with a list of rows to be INSERTed, the list will contain
          one row per row inserted in the statement, however it will contain
          ``None`` for any server-generated values.

        Future releases of SQLAlchemy will further generalize the
        "fast execution helper" feature of psycopg2 to suit other dialects,
        thus allowing this accessor to be of more general use.

        .. versionadded:: 1.4

        .. seealso::

            :attr:`_engine.CursorResult.inserted_primary_key`

        """
    if not self.context.compiled:
        raise exc.InvalidRequestError('Statement is not a compiled expression construct.')
    elif not self.context.isinsert:
        raise exc.InvalidRequestError('Statement is not an insert() expression construct.')
    elif self.context._is_explicit_returning:
        raise exc.InvalidRequestError("Can't call inserted_primary_key when returning() is used.")
    return self.context.inserted_primary_key_rows