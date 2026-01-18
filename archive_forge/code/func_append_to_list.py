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
def append_to_list(self, owner: RefCollection[_ET], list_: Deque[_ListenerFnType]) -> bool:
    if _stored_in_collection(self, owner):
        list_.append(self._listen_fn)
        return True
    else:
        return False