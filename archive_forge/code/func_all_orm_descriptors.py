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
@HasMemoized.memoized_attribute
def all_orm_descriptors(self) -> util.ReadOnlyProperties[InspectionAttr]:
    """A namespace of all :class:`.InspectionAttr` attributes associated
        with the mapped class.

        These attributes are in all cases Python :term:`descriptors`
        associated with the mapped class or its superclasses.

        This namespace includes attributes that are mapped to the class
        as well as attributes declared by extension modules.
        It includes any Python descriptor type that inherits from
        :class:`.InspectionAttr`.  This includes
        :class:`.QueryableAttribute`, as well as extension types such as
        :class:`.hybrid_property`, :class:`.hybrid_method` and
        :class:`.AssociationProxy`.

        To distinguish between mapped attributes and extension attributes,
        the attribute :attr:`.InspectionAttr.extension_type` will refer
        to a constant that distinguishes between different extension types.

        The sorting of the attributes is based on the following rules:

        1. Iterate through the class and its superclasses in order from
           subclass to superclass (i.e. iterate through ``cls.__mro__``)

        2. For each class, yield the attributes in the order in which they
           appear in ``__dict__``, with the exception of those in step
           3 below.  In Python 3.6 and above this ordering will be the
           same as that of the class' construction, with the exception
           of attributes that were added after the fact by the application
           or the mapper.

        3. If a certain attribute key is also in the superclass ``__dict__``,
           then it's included in the iteration for that class, and not the
           class in which it first appeared.

        The above process produces an ordering that is deterministic in terms
        of the order in which attributes were assigned to the class.

        .. versionchanged:: 1.3.19 ensured deterministic ordering for
           :meth:`_orm.Mapper.all_orm_descriptors`.

        When dealing with a :class:`.QueryableAttribute`, the
        :attr:`.QueryableAttribute.property` attribute refers to the
        :class:`.MapperProperty` property, which is what you get when
        referring to the collection of mapped properties via
        :attr:`_orm.Mapper.attrs`.

        .. warning::

            The :attr:`_orm.Mapper.all_orm_descriptors`
            accessor namespace is an
            instance of :class:`.OrderedProperties`.  This is
            a dictionary-like object which includes a small number of
            named methods such as :meth:`.OrderedProperties.items`
            and :meth:`.OrderedProperties.values`.  When
            accessing attributes dynamically, favor using the dict-access
            scheme, e.g. ``mapper.all_orm_descriptors[somename]`` over
            ``getattr(mapper.all_orm_descriptors, somename)`` to avoid name
            collisions.

        .. seealso::

            :attr:`_orm.Mapper.attrs`

        """
    return util.ReadOnlyProperties(dict(self.class_manager._all_sqla_attributes()))