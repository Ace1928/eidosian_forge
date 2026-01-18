from __future__ import annotations
import collections
from itertools import chain
import threading
from types import TracebackType
import typing
from typing import Any
from typing import cast
from typing import Collection
from typing import Deque
from typing import FrozenSet
from typing import Generic
from typing import Iterator
from typing import MutableMapping
from typing import MutableSequence
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import weakref
from . import legacy
from . import registry
from .registry import _ET
from .registry import _EventKey
from .registry import _ListenerFnType
from .. import exc
from .. import util
from ..util.concurrency import AsyncAdaptedLock
from ..util.typing import Protocol
def exec_once_unless_exception(self, *args: Any, **kw: Any) -> None:
    """Execute this event, but only if it has not been
        executed already for this collection, or was called
        by a previous exec_once_unless_exception call and
        raised an exception.

        If exec_once was already called, then this method will never run
        the callable regardless of whether it raised or not.

        .. versionadded:: 1.3.8

        """
    if not self._exec_once:
        self._exec_once_impl(True, *args, **kw)