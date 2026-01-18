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
def autoincrement_column(self) -> Optional[Column[int]]:
    """Returns the :class:`.Column` object which currently represents
        the "auto increment" column, if any, else returns None.

        This is based on the rules for :class:`.Column` as defined by the
        :paramref:`.Column.autoincrement` parameter, which generally means the
        column within a single integer column primary key constraint that is
        not constrained by a foreign key.   If the table does not have such
        a primary key constraint, then there's no "autoincrement" column.
        A :class:`.Table` may have only one column defined as the
        "autoincrement" column.

        .. versionadded:: 2.0.4

        .. seealso::

            :paramref:`.Column.autoincrement`

        """
    return self._autoincrement_column