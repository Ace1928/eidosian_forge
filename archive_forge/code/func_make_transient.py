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
def make_transient(instance: object) -> None:
    """Alter the state of the given instance so that it is :term:`transient`.

    .. note::

        :func:`.make_transient` is a special-case function for
        advanced use cases only.

    The given mapped instance is assumed to be in the :term:`persistent` or
    :term:`detached` state.   The function will remove its association with any
    :class:`.Session` as well as its :attr:`.InstanceState.identity`. The
    effect is that the object will behave as though it were newly constructed,
    except retaining any attribute / collection values that were loaded at the
    time of the call.   The :attr:`.InstanceState.deleted` flag is also reset
    if this object had been deleted as a result of using
    :meth:`.Session.delete`.

    .. warning::

        :func:`.make_transient` does **not** "unexpire" or otherwise eagerly
        load ORM-mapped attributes that are not currently loaded at the time
        the function is called.   This includes attributes which:

        * were expired via :meth:`.Session.expire`

        * were expired as the natural effect of committing a session
          transaction, e.g. :meth:`.Session.commit`

        * are normally :term:`lazy loaded` but are not currently loaded

        * are "deferred" (see :ref:`orm_queryguide_column_deferral`) and are
          not yet loaded

        * were not present in the query which loaded this object, such as that
          which is common in joined table inheritance and other scenarios.

        After :func:`.make_transient` is called, unloaded attributes such
        as those above will normally resolve to the value ``None`` when
        accessed, or an empty collection for a collection-oriented attribute.
        As the object is transient and un-associated with any database
        identity, it will no longer retrieve these values.

    .. seealso::

        :func:`.make_transient_to_detached`

    """
    state = attributes.instance_state(instance)
    s = _state_session(state)
    if s:
        s._expunge_states([state])
    state.expired_attributes.clear()
    if state.callables:
        del state.callables
    if state.key:
        del state.key
    if state._deleted:
        del state._deleted