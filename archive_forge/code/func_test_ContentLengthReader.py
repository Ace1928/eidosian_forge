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
def test_ContentLengthReader() -> None:
    t_body_reader(lambda: ContentLengthReader(0), b'', [EndOfMessage()])
    t_body_reader(lambda: ContentLengthReader(10), b'0123456789', [Data(data=b'0123456789'), EndOfMessage()])