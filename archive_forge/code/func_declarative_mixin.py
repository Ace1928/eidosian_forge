from __future__ import annotations
import itertools
import re
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import clsregistry
from . import instrumentation
from . import interfaces
from . import mapperlib
from ._orm_constructors import composite
from ._orm_constructors import deferred
from ._orm_constructors import mapped_column
from ._orm_constructors import relationship
from ._orm_constructors import synonym
from .attributes import InstrumentedAttribute
from .base import _inspect_mapped_class
from .base import _is_mapped_class
from .base import Mapped
from .base import ORMDescriptor
from .decl_base import _add_attribute
from .decl_base import _as_declarative
from .decl_base import _ClassScanMapperConfig
from .decl_base import _declarative_constructor
from .decl_base import _DeferredMapperConfig
from .decl_base import _del_attribute
from .decl_base import _mapper
from .descriptor_props import Composite
from .descriptor_props import Synonym
from .descriptor_props import Synonym as _orm_synonym
from .mapper import Mapper
from .properties import MappedColumn
from .relationships import RelationshipProperty
from .state import InstanceState
from .. import exc
from .. import inspection
from .. import util
from ..sql import sqltypes
from ..sql.base import _NoArg
from ..sql.elements import SQLCoreOperations
from ..sql.schema import MetaData
from ..sql.selectable import FromClause
from ..util import hybridmethod
from ..util import hybridproperty
from ..util import typing as compat_typing
from ..util.typing import CallableReference
from ..util.typing import flatten_newtype
from ..util.typing import is_generic
from ..util.typing import is_literal
from ..util.typing import is_newtype
from ..util.typing import is_pep695
from ..util.typing import Literal
from ..util.typing import Self
def declarative_mixin(cls: Type[_T]) -> Type[_T]:
    """Mark a class as providing the feature of "declarative mixin".

    E.g.::

        from sqlalchemy.orm import declared_attr
        from sqlalchemy.orm import declarative_mixin

        @declarative_mixin
        class MyMixin:

            @declared_attr
            def __tablename__(cls):
                return cls.__name__.lower()

            __table_args__ = {'mysql_engine': 'InnoDB'}
            __mapper_args__= {'always_refresh': True}

            id =  Column(Integer, primary_key=True)

        class MyModel(MyMixin, Base):
            name = Column(String(1000))

    The :func:`_orm.declarative_mixin` decorator currently does not modify
    the given class in any way; it's current purpose is strictly to assist
    the :ref:`Mypy plugin <mypy_toplevel>` in being able to identify
    SQLAlchemy declarative mixin classes when no other context is present.

    .. versionadded:: 1.4.6

    .. seealso::

        :ref:`orm_mixins_toplevel`

        :ref:`mypy_declarative_mixins` - in the
        :ref:`Mypy plugin documentation <mypy_toplevel>`

    """
    return cls