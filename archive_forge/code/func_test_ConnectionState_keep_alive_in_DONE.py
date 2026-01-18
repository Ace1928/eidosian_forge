import pytest
from .._events import (
from .._state import (
from .._util import LocalProtocolError
def test_ConnectionState_keep_alive_in_DONE() -> None:
    cs = ConnectionState()
    cs.process_event(CLIENT, Request)
    cs.process_event(CLIENT, EndOfMessage)
    assert cs.states[CLIENT] is DONE
    cs.process_keep_alive_disabled()
    assert cs.states[CLIENT] is MUST_CLOSE