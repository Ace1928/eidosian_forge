from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
def test_automatic_transfer_encoding_in_response() -> None:
    for user_headers in [[('Transfer-Encoding', 'chunked')], [], [('Transfer-Encoding', 'chunked'), ('Content-Length', '100')]]:
        user_headers = cast(List[Tuple[str, str]], user_headers)
        p = ConnectionPair()
        p.send(CLIENT, [Request(method='GET', target='/', headers=[('Host', 'example.com')]), EndOfMessage()])
        p.send(SERVER, Response(status_code=200, headers=user_headers), expect=Response(status_code=200, headers=[('Transfer-Encoding', 'chunked')]))
        c = Connection(SERVER)
        receive_and_get(c, b'GET / HTTP/1.0\r\n\r\n')
        assert c.send(Response(status_code=200, headers=user_headers)) == b'HTTP/1.1 200 \r\nConnection: close\r\n\r\n'
        assert c.send(Data(data=b'12345')) == b'12345'