from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import Optional
from typing import TypeVar
from .. import exc
from .. import util
from ..util._has_cy import HAS_CYEXTENSION
from ..util.typing import Protocol
from ..util.typing import Self
def connection_memoize(key: str) -> Callable[[_C], _C]:
    """Decorator, memoize a function in a connection.info stash.

    Only applicable to functions which take no arguments other than a
    connection.  The memo will be stored in ``connection.info[key]``.
    """

    @util.decorator
    def decorated(fn, self, connection):
        connection = connection.connect()
        try:
            return connection.info[key]
        except KeyError:
            connection.info[key] = val = fn(self, connection)
            return val
    return decorated