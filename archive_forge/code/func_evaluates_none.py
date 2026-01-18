from __future__ import annotations
from enum import Enum
from types import ModuleType
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Mapping
from typing import NewType
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .base import SchemaEventTarget
from .cache_key import CacheConst
from .cache_key import NO_CACHE
from .operators import ColumnOperators
from .visitors import Visitable
from .. import exc
from .. import util
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypeAliasType
from ..util.typing import TypedDict
from ..util.typing import TypeGuard
def evaluates_none(self) -> Self:
    """Return a copy of this type which has the
        :attr:`.should_evaluate_none` flag set to True.

        E.g.::

                Table(
                    'some_table', metadata,
                    Column(
                        String(50).evaluates_none(),
                        nullable=True,
                        server_default='no value')
                )

        The ORM uses this flag to indicate that a positive value of ``None``
        is passed to the column in an INSERT statement, rather than omitting
        the column from the INSERT statement which has the effect of firing
        off column-level defaults.   It also allows for types which have
        special behavior associated with the Python None value to indicate
        that the value doesn't necessarily translate into SQL NULL; a
        prime example of this is a JSON type which may wish to persist the
        JSON value ``'null'``.

        In all cases, the actual NULL SQL value can be always be
        persisted in any column by using
        the :obj:`_expression.null` SQL construct in an INSERT statement
        or associated with an ORM-mapped attribute.

        .. note::

            The "evaluates none" flag does **not** apply to a value
            of ``None`` passed to :paramref:`_schema.Column.default` or
            :paramref:`_schema.Column.server_default`; in these cases,
            ``None``
            still means "no default".

        .. seealso::

            :ref:`session_forcing_null` - in the ORM documentation

            :paramref:`.postgresql.JSON.none_as_null` - PostgreSQL JSON
            interaction with this flag.

            :attr:`.TypeEngine.should_evaluate_none` - class-level flag

        """
    typ = self.copy()
    typ.should_evaluate_none = True
    return typ