from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
def test_close_different_states() -> None:
    req = [Request(method='GET', target='/foo', headers=[('Host', 'a')]), EndOfMessage()]
    resp = [Response(status_code=200, headers=[(b'transfer-encoding', b'chunked')]), EndOfMessage()]
    p = ConnectionPair()
    p.send(CLIENT, ConnectionClosed())
    for conn in p.conns:
        assert conn.states == {CLIENT: CLOSED, SERVER: MUST_CLOSE}
    p = ConnectionPair()
    p.send(CLIENT, req)
    p.send(CLIENT, ConnectionClosed())
    for conn in p.conns:
        assert conn.states == {CLIENT: CLOSED, SERVER: SEND_RESPONSE}
    p = ConnectionPair()
    p.send(CLIENT, req)
    with pytest.raises(LocalProtocolError):
        p.conn[SERVER].send(ConnectionClosed())
    p.conn[CLIENT].receive_data(b'')
    with pytest.raises(RemoteProtocolError):
        p.conn[CLIENT].next_event()
    p = ConnectionPair()
    p.send(CLIENT, req)
    p.send(SERVER, resp)
    p.send(SERVER, ConnectionClosed())
    for conn in p.conns:
        assert conn.states == {CLIENT: MUST_CLOSE, SERVER: CLOSED}
    p = ConnectionPair()
    p.send(CLIENT, req)
    p.send(SERVER, resp)
    p.send(CLIENT, ConnectionClosed())
    p.send(SERVER, ConnectionClosed())
    p.send(CLIENT, ConnectionClosed())
    p.send(SERVER, ConnectionClosed())
    p = ConnectionPair()
    p.send(CLIENT, Request(method='GET', target='/', headers=[('Host', 'a'), ('Content-Length', '10')]))
    with pytest.raises(LocalProtocolError):
        p.conn[CLIENT].send(ConnectionClosed())
    p.conn[SERVER].receive_data(b'')
    with pytest.raises(RemoteProtocolError):
        p.conn[SERVER].next_event()