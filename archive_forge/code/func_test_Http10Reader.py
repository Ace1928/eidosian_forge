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
def test_Http10Reader() -> None:
    t_body_reader(Http10Reader, b'', [EndOfMessage()], do_eof=True)
    t_body_reader(Http10Reader, b'asdf', [Data(data=b'asdf')], do_eof=False)
    t_body_reader(Http10Reader, b'asdf', [Data(data=b'asdf'), EndOfMessage()], do_eof=True)