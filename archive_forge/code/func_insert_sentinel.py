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
def insert_sentinel(name: Optional[str]=None, type_: Optional[_TypeEngineArgument[_T]]=None, *, default: Optional[Any]=None, omit_from_statements: bool=True) -> Column[Any]:
    """Provides a surrogate :class:`_schema.Column` that will act as a
    dedicated insert :term:`sentinel` column, allowing efficient bulk
    inserts with deterministic RETURNING sorting for tables that
    don't otherwise have qualifying primary key configurations.

    Adding this column to a :class:`.Table` object requires that a
    corresponding database table actually has this column present, so if adding
    it to an existing model, existing database tables would need to be migrated
    (e.g. using ALTER TABLE or similar) to include this column.

    For background on how this object is used, see the section
    :ref:`engine_insertmanyvalues_sentinel_columns` as part of the
    section :ref:`engine_insertmanyvalues`.

    The :class:`_schema.Column` returned will be a nullable integer column by
    default and make use of a sentinel-specific default generator used only in
    "insertmanyvalues" operations.

    .. seealso::

        :func:`_orm.orm_insert_sentinel`

        :paramref:`_schema.Column.insert_sentinel`

        :ref:`engine_insertmanyvalues`

        :ref:`engine_insertmanyvalues_sentinel_columns`


    .. versionadded:: 2.0.10

    """
    return Column(name=name, type_=type_api.INTEGERTYPE if type_ is None else type_, default=default if default is not None else _InsertSentinelColumnDefault(), _omit_from_statements=omit_from_statements, insert_sentinel=True)