from __future__ import annotations
from typing import Any
from typing import Collection
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.sql import bindparam
from . import attributes
from . import interfaces
from . import relationships
from . import strategies
from .base import NEVER_SET
from .base import object_mapper
from .base import PassiveFlag
from .base import RelationshipDirection
from .. import exc
from .. import inspect
from .. import log
from .. import util
from ..sql import delete
from ..sql import insert
from ..sql import select
from ..sql import update
from ..sql.dml import Delete
from ..sql.dml import Insert
from ..sql.dml import Update
from ..util.typing import Literal
def fire_append_event(self, state: InstanceState[Any], dict_: _InstanceDict, value: Any, initiator: Optional[AttributeEventToken], collection_history: Optional[WriteOnlyHistory[Any]]=None) -> None:
    if collection_history is None:
        collection_history = self._modified_event(state, dict_)
    collection_history.add_added(value)
    for fn in self.dispatch.append:
        value = fn(state, value, initiator or self._append_token)
    if self.trackparent and value is not None:
        self.sethasparent(attributes.instance_state(value), state, True)