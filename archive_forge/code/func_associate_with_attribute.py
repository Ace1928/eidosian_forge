from within the mutable extension::
from __future__ import annotations
from collections import defaultdict
from typing import AbstractSet
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from weakref import WeakKeyDictionary
from .. import event
from .. import inspect
from .. import types
from .. import util
from ..orm import Mapper
from ..orm._typing import _ExternalEntityType
from ..orm._typing import _O
from ..orm._typing import _T
from ..orm.attributes import AttributeEventToken
from ..orm.attributes import flag_modified
from ..orm.attributes import InstrumentedAttribute
from ..orm.attributes import QueryableAttribute
from ..orm.context import QueryContext
from ..orm.decl_api import DeclarativeAttributeIntercept
from ..orm.state import InstanceState
from ..orm.unitofwork import UOWTransaction
from ..sql.base import SchemaEventTarget
from ..sql.schema import Column
from ..sql.type_api import TypeEngine
from ..util import memoized_property
from ..util.typing import SupportsIndex
from ..util.typing import TypeGuard
@classmethod
def associate_with_attribute(cls, attribute: InstrumentedAttribute[_O]) -> None:
    """Establish this type as a mutation listener for the given
        mapped descriptor.

        """
    cls._listen_on_attribute(attribute, True, attribute.class_)