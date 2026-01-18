from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
def test_HEAD_framing_headers() -> None:

    def setup(method: bytes, http_version: bytes) -> Connection:
        c = Connection(SERVER)
        c.receive_data(method + b' / HTTP/' + http_version + b'\r\n' + b'Host: example.com\r\n\r\n')
        assert type(c.next_event()) is Request
        assert type(c.next_event()) is EndOfMessage
        return c
    for method in [b'GET', b'HEAD']:
        c = setup(method, b'1.1')
        assert c.send(Response(status_code=200, headers=[])) == b'HTTP/1.1 200 \r\nTransfer-Encoding: chunked\r\n\r\n'
        c = setup(method, b'1.0')
        assert c.send(Response(status_code=200, headers=[])) == b'HTTP/1.1 200 \r\nConnection: close\r\n\r\n'
        c = setup(method, b'1.1')
        assert c.send(Response(status_code=200, headers=[('Content-Length', '100'), ('Transfer-Encoding', 'chunked')])) == b'HTTP/1.1 200 \r\nTransfer-Encoding: chunked\r\n\r\n'