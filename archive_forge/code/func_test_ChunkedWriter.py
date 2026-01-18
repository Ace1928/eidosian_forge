from typing import Any, Callable, Generator, List
import pytest
from .._events import (
from .._headers import Headers, normalize_and_validate
from .._readers import (
from .._receivebuffer import ReceiveBuffer
from .._state import (
from .._util import LocalProtocolError
from .._writers import (
from .helpers import normalize_data_events
def test_ChunkedWriter() -> None:
    w = ChunkedWriter()
    assert dowrite(w, Data(data=b'aaa')) == b'3\r\naaa\r\n'
    assert dowrite(w, Data(data=b'a' * 20)) == b'14\r\n' + b'a' * 20 + b'\r\n'
    assert dowrite(w, Data(data=b'')) == b''
    assert dowrite(w, EndOfMessage()) == b'0\r\n\r\n'
    assert dowrite(w, EndOfMessage(headers=[('Etag', 'asdf'), ('a', 'b')])) == b'0\r\nEtag: asdf\r\na: b\r\n\r\n'