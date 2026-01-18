from __future__ import annotations
from abc import ABC
import collections
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence as _typing_Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import ddl
from . import roles
from . import type_api
from . import visitors
from .base import _DefaultDescriptionTuple
from .base import _NoneName
from .base import _SentinelColumnCharacterization
from .base import _SentinelDefaultCharacterization
from .base import DedupeColumnCollection
from .base import DialectKWArgs
from .base import Executable
from .base import SchemaEventTarget as SchemaEventTarget
from .coercions import _document_text_coercion
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import quoted_name
from .elements import TextClause
from .selectable import TableClause
from .type_api import to_instance
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized
from ..util.typing import Final
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypedDict
from ..util.typing import TypeGuard
@property
def sorted_tables(self) -> List[Table]:
    """Returns a list of :class:`_schema.Table` objects sorted in order of
        foreign key dependency.

        The sorting will place :class:`_schema.Table`
        objects that have dependencies
        first, before the dependencies themselves, representing the
        order in which they can be created.   To get the order in which
        the tables would be dropped, use the ``reversed()`` Python built-in.

        .. warning::

            The :attr:`.MetaData.sorted_tables` attribute cannot by itself
            accommodate automatic resolution of dependency cycles between
            tables, which are usually caused by mutually dependent foreign key
            constraints. When these cycles are detected, the foreign keys
            of these tables are omitted from consideration in the sort.
            A warning is emitted when this condition occurs, which will be an
            exception raise in a future release.   Tables which are not part
            of the cycle will still be returned in dependency order.

            To resolve these cycles, the
            :paramref:`_schema.ForeignKeyConstraint.use_alter` parameter may be
            applied to those constraints which create a cycle.  Alternatively,
            the :func:`_schema.sort_tables_and_constraints` function will
            automatically return foreign key constraints in a separate
            collection when cycles are detected so that they may be applied
            to a schema separately.

            .. versionchanged:: 1.3.17 - a warning is emitted when
               :attr:`.MetaData.sorted_tables` cannot perform a proper sort
               due to cyclical dependencies.  This will be an exception in a
               future release.  Additionally, the sort will continue to return
               other tables not involved in the cycle in dependency order which
               was not the case previously.

        .. seealso::

            :func:`_schema.sort_tables`

            :func:`_schema.sort_tables_and_constraints`

            :attr:`_schema.MetaData.tables`

            :meth:`_reflection.Inspector.get_table_names`

            :meth:`_reflection.Inspector.get_sorted_table_and_fkc_names`


        """
    return ddl.sort_tables(sorted(self.tables.values(), key=lambda t: t.key))