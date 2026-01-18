from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
def test_max_incomplete_event_size_countermeasure() -> None:
    c = Connection(SERVER)
    c.receive_data(b'GET / HTTP/1.0\r\nEndless: ')
    assert c.next_event() is NEED_DATA
    with pytest.raises(RemoteProtocolError):
        while True:
            c.receive_data(b'a' * 1024)
            c.next_event()
    c = Connection(SERVER, max_incomplete_event_size=5000)
    c.receive_data(b'GET / HTTP/1.0\r\nBig: ')
    c.receive_data(b'a' * 4000)
    c.receive_data(b'\r\n\r\n')
    assert get_all_events(c) == [Request(method='GET', target='/', http_version='1.0', headers=[('big', 'a' * 4000)]), EndOfMessage()]
    c = Connection(SERVER, max_incomplete_event_size=4000)
    c.receive_data(b'GET / HTTP/1.0\r\nBig: ')
    c.receive_data(b'a' * 4000)
    with pytest.raises(RemoteProtocolError):
        c.next_event()
    c = Connection(SERVER, max_incomplete_event_size=5000)
    c.receive_data(b'GET / HTTP/1.0\r\nContent-Length: 10000')
    c.receive_data(b'\r\n\r\n' + b'a' * 10000)
    assert get_all_events(c) == [Request(method='GET', target='/', http_version='1.0', headers=[('Content-Length', '10000')]), Data(data=b'a' * 10000), EndOfMessage()]
    c = Connection(SERVER, max_incomplete_event_size=100)
    c.receive_data(b'GET /1 HTTP/1.1\r\nHost: a\r\n\r\nGET /2 HTTP/1.1\r\nHost: b\r\n\r\n' + b'X' * 1000)
    assert get_all_events(c) == [Request(method='GET', target='/1', headers=[('host', 'a')]), EndOfMessage()]
    c.receive_data(b'X' * 1000)
    c.send(Response(status_code=200, headers=[]))
    c.send(EndOfMessage())
    c.start_next_cycle()
    assert get_all_events(c) == [Request(method='GET', target='/2', headers=[('host', 'b')]), EndOfMessage()]
    c.send(Response(status_code=200, headers=[]))
    c.send(EndOfMessage())
    c.start_next_cycle()
    with pytest.raises(RemoteProtocolError):
        c.next_event()