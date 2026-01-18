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
def install_member(self, key: str, implementation: Any) -> None:
    if key in (self.STATE_ATTR, self.MANAGER_ATTR):
        raise KeyError('%r: requested attribute name conflicts with instrumentation attribute of the same name.' % key)
    self.originals.setdefault(key, self.class_.__dict__.get(key, DEL_ATTR))
    setattr(self.class_, key, implementation)