from __future__ import annotations
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
import weakref
from . import util as orm_util
from .. import exc as sa_exc
def contains_state(self, state: InstanceState[Any]) -> bool:
    if state.key in self._dict:
        if TYPE_CHECKING:
            assert state.key is not None
        try:
            return self._dict[state.key] is state
        except KeyError:
            return False
    else:
        return False