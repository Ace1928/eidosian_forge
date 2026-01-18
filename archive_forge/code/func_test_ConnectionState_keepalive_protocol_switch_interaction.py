import pytest
from .._events import (
from .._state import (
from .._util import LocalProtocolError
def test_ConnectionState_keepalive_protocol_switch_interaction() -> None:
    cs = ConnectionState()
    cs.process_client_switch_proposal(_SWITCH_UPGRADE)
    cs.process_event(CLIENT, Request)
    cs.process_keep_alive_disabled()
    cs.process_event(CLIENT, Data)
    assert cs.states == {CLIENT: SEND_BODY, SERVER: SEND_RESPONSE}
    cs.process_event(CLIENT, EndOfMessage)
    assert cs.states == {CLIENT: MIGHT_SWITCH_PROTOCOL, SERVER: SEND_RESPONSE}
    cs.process_event(SERVER, Response)
    assert cs.states == {CLIENT: MUST_CLOSE, SERVER: SEND_BODY}