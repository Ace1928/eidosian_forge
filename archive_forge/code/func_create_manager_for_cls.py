from __future__ import annotations
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import base
from . import collections
from . import exc
from . import interfaces
from . import state
from ._typing import _O
from .attributes import _is_collection_attribute_impl
from .. import util
from ..event import EventTarget
from ..util import HasMemoized
from ..util.typing import Literal
from ..util.typing import Protocol
def create_manager_for_cls(self, class_: Type[_O]) -> ClassManager[_O]:
    assert class_ is not None
    assert opt_manager_of_class(class_) is None
    manager, factory = self._locate_extended_factory(class_)
    if factory is None:
        factory = ClassManager
        manager = ClassManager(class_)
    else:
        assert manager is not None
    self._check_conflicts(class_, factory)
    manager.factory = factory
    return manager