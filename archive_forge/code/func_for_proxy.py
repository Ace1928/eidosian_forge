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
@classmethod
def for_proxy(cls, parent: AssociationProxy[_T], owning_class: Type[Any], parent_instance: Any) -> AssociationProxyInstance[_T]:
    target_collection = parent.target_collection
    value_attr = parent.value_attr
    prop = cast('orm.RelationshipProperty[_T]', orm.class_mapper(owning_class).get_property(target_collection))
    if not isinstance(prop, orm.RelationshipProperty):
        raise NotImplementedError('association proxy to a non-relationship intermediary is not supported') from None
    target_class = prop.mapper.class_
    try:
        target_assoc = cast('AssociationProxyInstance[_T]', cls._cls_unwrap_target_assoc_proxy(target_class, value_attr))
    except AttributeError:
        return AmbiguousAssociationProxyInstance(parent, owning_class, target_class, value_attr)
    except Exception as err:
        raise exc.InvalidRequestError(f'Association proxy received an unexpected error when trying to retreive attribute "{target_class.__name__}.{parent.value_attr}" from class "{target_class.__name__}": {err}') from err
    else:
        return cls._construct_for_assoc(target_assoc, parent, owning_class, target_class, value_attr)