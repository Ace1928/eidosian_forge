from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
def test__keep_alive() -> None:
    assert _keep_alive(Request(method='GET', target='/', headers=[('Host', 'Example.com')]))
    assert not _keep_alive(Request(method='GET', target='/', headers=[('Host', 'Example.com'), ('Connection', 'close')]))
    assert not _keep_alive(Request(method='GET', target='/', headers=[('Host', 'Example.com'), ('Connection', 'a, b, cLOse, foo')]))
    assert not _keep_alive(Request(method='GET', target='/', headers=[], http_version='1.0'))
    assert _keep_alive(Response(status_code=200, headers=[]))
    assert not _keep_alive(Response(status_code=200, headers=[('Connection', 'close')]))
    assert not _keep_alive(Response(status_code=200, headers=[('Connection', 'a, b, cLOse, foo')]))
    assert not _keep_alive(Response(status_code=200, headers=[], http_version='1.0'))