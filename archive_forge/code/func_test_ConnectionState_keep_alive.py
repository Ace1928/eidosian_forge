import pytest
from .._events import (
from .._state import (
from .._util import LocalProtocolError
def test_ConnectionState_keep_alive() -> None:
    cs = ConnectionState()
    cs.process_event(CLIENT, Request)
    cs.process_keep_alive_disabled()
    cs.process_event(CLIENT, EndOfMessage)
    assert cs.states == {CLIENT: MUST_CLOSE, SERVER: SEND_RESPONSE}
    cs.process_event(SERVER, Response)
    cs.process_event(SERVER, EndOfMessage)
    assert cs.states == {CLIENT: MUST_CLOSE, SERVER: MUST_CLOSE}