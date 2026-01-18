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
def instrument_attribute(self, key: str, inst: QueryableAttribute[Any], propagated: bool=False) -> None:
    if propagated:
        if key in self.local_attrs:
            return
    else:
        self.local_attrs[key] = inst
        self.install_descriptor(key, inst)
    self._reset_memoizations()
    self[key] = inst
    for cls in self.class_.__subclasses__():
        manager = self._subclass_manager(cls)
        manager.instrument_attribute(key, inst, True)