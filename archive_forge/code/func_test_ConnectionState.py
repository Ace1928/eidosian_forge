import pytest
from .._events import (
from .._state import (
from .._util import LocalProtocolError
def test_ConnectionState() -> None:
    cs = ConnectionState()
    assert cs.states == {CLIENT: IDLE, SERVER: IDLE}
    cs.process_event(CLIENT, Request)
    assert cs.states == {CLIENT: SEND_BODY, SERVER: SEND_RESPONSE}
    with pytest.raises(LocalProtocolError):
        cs.process_event(CLIENT, Request)
    assert cs.states == {CLIENT: SEND_BODY, SERVER: SEND_RESPONSE}
    cs.process_event(SERVER, InformationalResponse)
    assert cs.states == {CLIENT: SEND_BODY, SERVER: SEND_RESPONSE}
    cs.process_event(SERVER, Response)
    assert cs.states == {CLIENT: SEND_BODY, SERVER: SEND_BODY}
    cs.process_event(CLIENT, EndOfMessage)
    cs.process_event(SERVER, EndOfMessage)
    assert cs.states == {CLIENT: DONE, SERVER: DONE}
    cs.process_event(SERVER, ConnectionClosed)
    assert cs.states == {CLIENT: MUST_CLOSE, SERVER: CLOSED}