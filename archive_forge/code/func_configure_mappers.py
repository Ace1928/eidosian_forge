from __future__ import annotations
from collections import deque
from functools import reduce
from itertools import chain
import sys
import threading
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Deque
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import exc as orm_exc
from . import instrumentation
from . import loading
from . import properties
from . import util as orm_util
from ._typing import _O
from .base import _class_to_mapper
from .base import _parse_mapper_argument
from .base import _state_mapper
from .base import PassiveFlag
from .base import state_str
from .interfaces import _MappedAttribute
from .interfaces import EXT_SKIP
from .interfaces import InspectionAttr
from .interfaces import MapperProperty
from .interfaces import ORMEntityColumnsClauseRole
from .interfaces import ORMFromClauseRole
from .interfaces import StrategizedProperty
from .path_registry import PathRegistry
from .. import event
from .. import exc as sa_exc
from .. import inspection
from .. import log
from .. import schema
from .. import sql
from .. import util
from ..event import dispatcher
from ..event import EventTarget
from ..sql import base as sql_base
from ..sql import coercions
from ..sql import expression
from ..sql import operators
from ..sql import roles
from ..sql import TableClause
from ..sql import util as sql_util
from ..sql import visitors
from ..sql.cache_key import MemoizedHasCacheKey
from ..sql.elements import KeyedColumnElement
from ..sql.schema import Column
from ..sql.schema import Table
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import HasMemoized
from ..util import HasMemoized_ro_memoized_attribute
from ..util.typing import Literal
def configure_mappers() -> None:
    """Initialize the inter-mapper relationships of all mappers that
    have been constructed thus far across all :class:`_orm.registry`
    collections.

    The configure step is used to reconcile and initialize the
    :func:`_orm.relationship` linkages between mapped classes, as well as to
    invoke configuration events such as the
    :meth:`_orm.MapperEvents.before_configured` and
    :meth:`_orm.MapperEvents.after_configured`, which may be used by ORM
    extensions or user-defined extension hooks.

    Mapper configuration is normally invoked automatically, the first time
    mappings from a particular :class:`_orm.registry` are used, as well as
    whenever mappings are used and additional not-yet-configured mappers have
    been constructed. The automatic configuration process however is local only
    to the :class:`_orm.registry` involving the target mapper and any related
    :class:`_orm.registry` objects which it may depend on; this is
    equivalent to invoking the :meth:`_orm.registry.configure` method
    on a particular :class:`_orm.registry`.

    By contrast, the :func:`_orm.configure_mappers` function will invoke the
    configuration process on all :class:`_orm.registry` objects that
    exist in memory, and may be useful for scenarios where many individual
    :class:`_orm.registry` objects that are nonetheless interrelated are
    in use.

    .. versionchanged:: 1.4

        As of SQLAlchemy 1.4.0b2, this function works on a
        per-:class:`_orm.registry` basis, locating all :class:`_orm.registry`
        objects present and invoking the :meth:`_orm.registry.configure` method
        on each. The :meth:`_orm.registry.configure` method may be preferred to
        limit the configuration of mappers to those local to a particular
        :class:`_orm.registry` and/or declarative base class.

    Points at which automatic configuration is invoked include when a mapped
    class is instantiated into an instance, as well as when ORM queries
    are emitted using :meth:`.Session.query` or :meth:`_orm.Session.execute`
    with an ORM-enabled statement.

    The mapper configure process, whether invoked by
    :func:`_orm.configure_mappers` or from :meth:`_orm.registry.configure`,
    provides several event hooks that can be used to augment the mapper
    configuration step. These hooks include:

    * :meth:`.MapperEvents.before_configured` - called once before
      :func:`.configure_mappers` or :meth:`_orm.registry.configure` does any
      work; this can be used to establish additional options, properties, or
      related mappings before the operation proceeds.

    * :meth:`.MapperEvents.mapper_configured` - called as each individual
      :class:`_orm.Mapper` is configured within the process; will include all
      mapper state except for backrefs set up by other mappers that are still
      to be configured.

    * :meth:`.MapperEvents.after_configured` - called once after
      :func:`.configure_mappers` or :meth:`_orm.registry.configure` is
      complete; at this stage, all :class:`_orm.Mapper` objects that fall
      within the scope of the configuration operation will be fully configured.
      Note that the calling application may still have other mappings that
      haven't been produced yet, such as if they are in modules as yet
      unimported, and may also have mappings that are still to be configured,
      if they are in other :class:`_orm.registry` collections not part of the
      current scope of configuration.

    """
    _configure_registries(_all_registries(), cascade=True)