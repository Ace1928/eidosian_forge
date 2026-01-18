from typing import cast, List, Type, Union, ValuesView
from .._connection import Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import CLIENT, CLOSED, DONE, MUST_CLOSE, SERVER
from .._util import Sentinel
def get_all_events(conn: Connection) -> List[Event]:
    got_events = []
    while True:
        event = conn.next_event()
        if event in (NEED_DATA, PAUSED):
            break
        event = cast(Event, event)
        got_events.append(event)
        if type(event) is ConnectionClosed:
            break
    return got_events