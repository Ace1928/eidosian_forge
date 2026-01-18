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
def ddl_if(self, dialect: Optional[str]=None, callable_: Optional[ddl.DDLIfCallable]=None, state: Optional[Any]=None) -> Self:
    """apply a conditional DDL rule to this schema item.

        These rules work in a similar manner to the
        :meth:`.ExecutableDDLElement.execute_if` callable, with the added
        feature that the criteria may be checked within the DDL compilation
        phase for a construct such as :class:`.CreateTable`.
        :meth:`.HasConditionalDDL.ddl_if` currently applies towards the
        :class:`.Index` construct as well as all :class:`.Constraint`
        constructs.

        :param dialect: string name of a dialect, or a tuple of string names
         to indicate multiple dialect types.

        :param callable\\_: a callable that is constructed using the same form
         as that described in
         :paramref:`.ExecutableDDLElement.execute_if.callable_`.

        :param state: any arbitrary object that will be passed to the
         callable, if present.

        .. versionadded:: 2.0

        .. seealso::

            :ref:`schema_ddl_ddl_if` - background and usage examples


        """
    self._ddl_if = ddl.DDLIf(dialect, callable_, state)
    return self