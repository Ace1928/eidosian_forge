from __future__ import annotations
import collections
import types
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Deque
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
import weakref
from .. import exc
from .. import util
def remove_from_list(self, owner: RefCollection[_ET], list_: Deque[_ListenerFnType]) -> None:
    _removed_from_collection(self, owner)
    list_.remove(self._listen_fn)