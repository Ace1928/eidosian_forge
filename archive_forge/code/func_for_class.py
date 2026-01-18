from __future__ import annotations
import operator
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Generic
from typing import ItemsView
from typing import Iterable
from typing import Iterator
from typing import KeysView
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import MutableSequence
from typing import MutableSet
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from typing import ValuesView
from .. import ColumnElement
from .. import exc
from .. import inspect
from .. import orm
from .. import util
from ..orm import collections
from ..orm import InspectionAttrExtensionType
from ..orm import interfaces
from ..orm import ORMDescriptor
from ..orm.base import SQLORMOperations
from ..orm.interfaces import _AttributeOptions
from ..orm.interfaces import _DCAttributeOptions
from ..orm.interfaces import _DEFAULT_ATTRIBUTE_OPTIONS
from ..sql import operators
from ..sql import or_
from ..sql.base import _NoArg
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import SupportsIndex
from ..util.typing import SupportsKeysAndGetItem
def for_class(self, class_: Type[Any], obj: Optional[object]=None) -> AssociationProxyInstance[_T]:
    """Return the internal state local to a specific mapped class.

        E.g., given a class ``User``::

            class User(Base):
                # ...

                keywords = association_proxy('kws', 'keyword')

        If we access this :class:`.AssociationProxy` from
        :attr:`_orm.Mapper.all_orm_descriptors`, and we want to view the
        target class for this proxy as mapped by ``User``::

            inspect(User).all_orm_descriptors["keywords"].for_class(User).target_class

        This returns an instance of :class:`.AssociationProxyInstance` that
        is specific to the ``User`` class.   The :class:`.AssociationProxy`
        object remains agnostic of its parent class.

        :param class\\_: the class that we are returning state for.

        :param obj: optional, an instance of the class that is required
         if the attribute refers to a polymorphic target, e.g. where we have
         to look at the type of the actual destination object to get the
         complete path.

        .. versionadded:: 1.3 - :class:`.AssociationProxy` no longer stores
           any state specific to a particular parent class; the state is now
           stored in per-class :class:`.AssociationProxyInstance` objects.


        """
    return self._as_instance(class_, obj)