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
def test_ChunkedReader() -> None:
    t_body_reader(ChunkedReader, b'0\r\n\r\n', [EndOfMessage()])
    t_body_reader(ChunkedReader, b'0\r\nSome: header\r\n\r\n', [EndOfMessage(headers=[('Some', 'header')])])
    t_body_reader(ChunkedReader, b'5\r\n01234\r\n' + b'10\r\n0123456789abcdef\r\n' + b'0\r\n' + b'Some: header\r\n\r\n', [Data(data=b'012340123456789abcdef'), EndOfMessage(headers=[('Some', 'header')])])
    t_body_reader(ChunkedReader, b'5\r\n01234\r\n' + b'10\r\n0123456789abcdef\r\n' + b'0\r\n\r\n', [Data(data=b'012340123456789abcdef'), EndOfMessage()])
    t_body_reader(ChunkedReader, b'aA\r\n' + b'x' * 170 + b'\r\n' + b'0\r\n\r\n', [Data(data=b'x' * 170), EndOfMessage()])
    with pytest.raises(LocalProtocolError):
        t_body_reader(ChunkedReader, b'9' * 100 + b'\r\nxxx', [Data(data=b'xxx')])
    with pytest.raises(LocalProtocolError):
        t_body_reader(ChunkedReader, b'10\x00\r\nxxx', None)
    t_body_reader(ChunkedReader, b'5; hello=there\r\n' + b'xxxxx' + b'\r\n' + b'0; random="junk"; some=more; canbe=lonnnnngg\r\n\r\n', [Data(data=b'xxxxx'), EndOfMessage()])
    t_body_reader(ChunkedReader, b'5   \t \r\n01234\r\n' + b'0\r\n\r\n', [Data(data=b'01234'), EndOfMessage()])