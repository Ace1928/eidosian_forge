from typing import cast, List, Type, Union, ValuesView
from .._connection import Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import CLIENT, CLOSED, DONE, MUST_CLOSE, SERVER
from .._util import Sentinel
@property
def conns(self) -> ValuesView[Connection]:
    return self.conn.values()