from __future__ import annotations
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import interfaces
from .descriptor_props import SynonymProperty
from .properties import ColumnProperty
from .util import class_mapper
from .. import exc
from .. import inspection
from .. import util
from ..sql.schema import _get_table_key
from ..util.typing import CallableReference
def remove_class(self, name: str, cls: Type[Any]) -> None:
    if name in self.contents:
        existing = cast(_MultipleClassMarker, self.contents[name])
        existing.remove_item(cls)