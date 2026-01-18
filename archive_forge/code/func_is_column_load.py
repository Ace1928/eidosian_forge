from __future__ import annotations
import contextlib
from enum import Enum
import itertools
import sys
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import bulk_persistence
from . import context
from . import descriptor_props
from . import exc
from . import identity
from . import loading
from . import query
from . import state as statelib
from ._typing import _O
from ._typing import insp_is_mapper
from ._typing import is_composite_class
from ._typing import is_orm_option
from ._typing import is_user_defined_option
from .base import _class_to_mapper
from .base import _none_set
from .base import _state_mapper
from .base import instance_str
from .base import LoaderCallableStatus
from .base import object_mapper
from .base import object_state
from .base import PassiveFlag
from .base import state_str
from .context import FromStatement
from .context import ORMCompileState
from .identity import IdentityMap
from .query import Query
from .state import InstanceState
from .state_changes import _StateChange
from .state_changes import _StateChangeState
from .state_changes import _StateChangeStates
from .unitofwork import UOWTransaction
from .. import engine
from .. import exc as sa_exc
from .. import sql
from .. import util
from ..engine import Connection
from ..engine import Engine
from ..engine.util import TransactionalContext
from ..event import dispatcher
from ..event import EventTarget
from ..inspection import inspect
from ..inspection import Inspectable
from ..sql import coercions
from ..sql import dml
from ..sql import roles
from ..sql import Select
from ..sql import TableClause
from ..sql import visitors
from ..sql.base import _NoArg
from ..sql.base import CompileState
from ..sql.schema import Table
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import IdentitySet
from ..util.typing import Literal
from ..util.typing import Protocol
@property
def is_column_load(self) -> bool:
    """Return True if the operation is refreshing column-oriented
        attributes on an existing ORM object.

        This occurs during operations such as :meth:`_orm.Session.refresh`,
        as well as when an attribute deferred by :func:`_orm.defer` is
        being loaded, or an attribute that was expired either directly
        by :meth:`_orm.Session.expire` or via a commit operation is being
        loaded.

        Handlers will very likely not want to add any options to queries
        when such an operation is occurring as the query should be a straight
        primary key fetch which should not have any additional WHERE criteria,
        and loader options travelling with the instance
        will have already been added to the query.

        .. versionadded:: 1.4.0b2

        .. seealso::

            :attr:`_orm.ORMExecuteState.is_relationship_load`

        """
    opts = self._orm_compile_options()
    return opts is not None and opts._for_refresh_state