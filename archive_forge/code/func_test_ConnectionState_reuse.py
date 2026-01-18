import pytest
from .._events import (
from .._state import (
from .._util import LocalProtocolError
def test_ConnectionState_reuse() -> None:
    cs = ConnectionState()
    with pytest.raises(LocalProtocolError):
        cs.start_next_cycle()
    cs.process_event(CLIENT, Request)
    cs.process_event(CLIENT, EndOfMessage)
    with pytest.raises(LocalProtocolError):
        cs.start_next_cycle()
    cs.process_event(SERVER, Response)
    cs.process_event(SERVER, EndOfMessage)
    cs.start_next_cycle()
    assert cs.states == {CLIENT: IDLE, SERVER: IDLE}
    cs.process_event(CLIENT, Request)
    cs.process_keep_alive_disabled()
    cs.process_event(CLIENT, EndOfMessage)
    cs.process_event(SERVER, Response)
    cs.process_event(SERVER, EndOfMessage)
    with pytest.raises(LocalProtocolError):
        cs.start_next_cycle()
    cs = ConnectionState()
    cs.process_event(CLIENT, Request)
    cs.process_event(CLIENT, EndOfMessage)
    cs.process_event(CLIENT, ConnectionClosed)
    cs.process_event(SERVER, Response)
    cs.process_event(SERVER, EndOfMessage)
    with pytest.raises(LocalProtocolError):
        cs.start_next_cycle()
    cs = ConnectionState()
    cs.process_client_switch_proposal(_SWITCH_UPGRADE)
    cs.process_event(CLIENT, Request)
    cs.process_event(CLIENT, EndOfMessage)
    cs.process_event(SERVER, InformationalResponse, _SWITCH_UPGRADE)
    with pytest.raises(LocalProtocolError):
        cs.start_next_cycle()
    cs = ConnectionState()
    cs.process_client_switch_proposal(_SWITCH_UPGRADE)
    cs.process_event(CLIENT, Request)
    cs.process_event(CLIENT, EndOfMessage)
    cs.process_event(SERVER, Response)
    cs.process_event(SERVER, EndOfMessage)
    cs.start_next_cycle()
    assert cs.states == {CLIENT: IDLE, SERVER: IDLE}