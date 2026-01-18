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
def association_proxy(target_collection: str, attr: str, *, creator: Optional[_CreatorProtocol]=None, getset_factory: Optional[_GetSetFactoryProtocol]=None, proxy_factory: Optional[_ProxyFactoryProtocol]=None, proxy_bulk_set: Optional[_ProxyBulkSetProtocol]=None, info: Optional[_InfoType]=None, cascade_scalar_deletes: bool=False, create_on_none_assignment: bool=False, init: Union[_NoArg, bool]=_NoArg.NO_ARG, repr: Union[_NoArg, bool]=_NoArg.NO_ARG, default: Optional[Any]=_NoArg.NO_ARG, default_factory: Union[_NoArg, Callable[[], _T]]=_NoArg.NO_ARG, compare: Union[_NoArg, bool]=_NoArg.NO_ARG, kw_only: Union[_NoArg, bool]=_NoArg.NO_ARG) -> AssociationProxy[Any]:
    """Return a Python property implementing a view of a target
    attribute which references an attribute on members of the
    target.

    The returned value is an instance of :class:`.AssociationProxy`.

    Implements a Python property representing a relationship as a collection
    of simpler values, or a scalar value.  The proxied property will mimic
    the collection type of the target (list, dict or set), or, in the case of
    a one to one relationship, a simple scalar value.

    :param target_collection: Name of the attribute that is the immediate
      target.  This attribute is typically mapped by
      :func:`~sqlalchemy.orm.relationship` to link to a target collection, but
      can also be a many-to-one or non-scalar relationship.

    :param attr: Attribute on the associated instance or instances that
      are available on instances of the target object.

    :param creator: optional.

      Defines custom behavior when new items are added to the proxied
      collection.

      By default, adding new items to the collection will trigger a
      construction of an instance of the target object, passing the given
      item as a positional argument to the target constructor.  For cases
      where this isn't sufficient, :paramref:`.association_proxy.creator`
      can supply a callable that will construct the object in the
      appropriate way, given the item that was passed.

      For list- and set- oriented collections, a single argument is
      passed to the callable. For dictionary oriented collections, two
      arguments are passed, corresponding to the key and value.

      The :paramref:`.association_proxy.creator` callable is also invoked
      for scalar (i.e. many-to-one, one-to-one) relationships. If the
      current value of the target relationship attribute is ``None``, the
      callable is used to construct a new object.  If an object value already
      exists, the given attribute value is populated onto that object.

      .. seealso::

        :ref:`associationproxy_creator`

    :param cascade_scalar_deletes: when True, indicates that setting
        the proxied value to ``None``, or deleting it via ``del``, should
        also remove the source object.  Only applies to scalar attributes.
        Normally, removing the proxied target will not remove the proxy
        source, as this object may have other state that is still to be
        kept.

        .. versionadded:: 1.3

        .. seealso::

            :ref:`cascade_scalar_deletes` - complete usage example

    :param create_on_none_assignment: when True, indicates that setting
      the proxied value to ``None`` should **create** the source object
      if it does not exist, using the creator.  Only applies to scalar
      attributes.  This is mutually exclusive
      vs. the :paramref:`.assocation_proxy.cascade_scalar_deletes`.

      .. versionadded:: 2.0.18

    :param init: Specific to :ref:`orm_declarative_native_dataclasses`,
     specifies if the mapped attribute should be part of the ``__init__()``
     method as generated by the dataclass process.

     .. versionadded:: 2.0.0b4

    :param repr: Specific to :ref:`orm_declarative_native_dataclasses`,
     specifies if the attribute established by this :class:`.AssociationProxy`
     should be part of the ``__repr__()`` method as generated by the dataclass
     process.

     .. versionadded:: 2.0.0b4

    :param default_factory: Specific to
     :ref:`orm_declarative_native_dataclasses`, specifies a default-value
     generation function that will take place as part of the ``__init__()``
     method as generated by the dataclass process.

     .. versionadded:: 2.0.0b4

    :param compare: Specific to
     :ref:`orm_declarative_native_dataclasses`, indicates if this field
     should be included in comparison operations when generating the
     ``__eq__()`` and ``__ne__()`` methods for the mapped class.

     .. versionadded:: 2.0.0b4

    :param kw_only: Specific to :ref:`orm_declarative_native_dataclasses`,
     indicates if this field should be marked as keyword-only when generating
     the ``__init__()`` method as generated by the dataclass process.

     .. versionadded:: 2.0.0b4

    :param info: optional, will be assigned to
     :attr:`.AssociationProxy.info` if present.


    The following additional parameters involve injection of custom behaviors
    within the :class:`.AssociationProxy` object and are for advanced use
    only:

    :param getset_factory: Optional.  Proxied attribute access is
        automatically handled by routines that get and set values based on
        the `attr` argument for this proxy.

        If you would like to customize this behavior, you may supply a
        `getset_factory` callable that produces a tuple of `getter` and
        `setter` functions.  The factory is called with two arguments, the
        abstract type of the underlying collection and this proxy instance.

    :param proxy_factory: Optional.  The type of collection to emulate is
        determined by sniffing the target collection.  If your collection
        type can't be determined by duck typing or you'd like to use a
        different collection implementation, you may supply a factory
        function to produce those collections.  Only applicable to
        non-scalar relationships.

    :param proxy_bulk_set: Optional, use with proxy_factory.


    """
    return AssociationProxy(target_collection, attr, creator=creator, getset_factory=getset_factory, proxy_factory=proxy_factory, proxy_bulk_set=proxy_bulk_set, info=info, cascade_scalar_deletes=cascade_scalar_deletes, create_on_none_assignment=create_on_none_assignment, attribute_options=_AttributeOptions(init, repr, default, default_factory, compare, kw_only))