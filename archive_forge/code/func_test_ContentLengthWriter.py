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
def test_ContentLengthWriter() -> None:
    w = ContentLengthWriter(5)
    assert dowrite(w, Data(data=b'123')) == b'123'
    assert dowrite(w, Data(data=b'45')) == b'45'
    assert dowrite(w, EndOfMessage()) == b''
    w = ContentLengthWriter(5)
    with pytest.raises(LocalProtocolError):
        dowrite(w, Data(data=b'123456'))
    w = ContentLengthWriter(5)
    dowrite(w, Data(data=b'123'))
    with pytest.raises(LocalProtocolError):
        dowrite(w, Data(data=b'456'))
    w = ContentLengthWriter(5)
    dowrite(w, Data(data=b'123'))
    with pytest.raises(LocalProtocolError):
        dowrite(w, EndOfMessage())
    w = ContentLengthWriter(5)
    dowrite(w, Data(data=b'123')) == b'123'
    dowrite(w, Data(data=b'45')) == b'45'
    with pytest.raises(LocalProtocolError):
        dowrite(w, EndOfMessage(headers=[('Etag', 'asdf')]))