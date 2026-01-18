import pytest
from .._events import (
from .._state import (
from .._util import LocalProtocolError
def test_ConnectionState_inconsistent_protocol_switch() -> None:
    for client_switches, server_switch in [([], _SWITCH_CONNECT), ([], _SWITCH_UPGRADE), ([_SWITCH_UPGRADE], _SWITCH_CONNECT), ([_SWITCH_CONNECT], _SWITCH_UPGRADE)]:
        cs = ConnectionState()
        for client_switch in client_switches:
            cs.process_client_switch_proposal(client_switch)
        cs.process_event(CLIENT, Request)
        with pytest.raises(LocalProtocolError):
            cs.process_event(SERVER, Response, server_switch)