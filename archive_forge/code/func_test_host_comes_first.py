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
def test_host_comes_first() -> None:
    tw(write_headers, normalize_and_validate([('foo', 'bar'), ('Host', 'example.com')]), b'Host: example.com\r\nfoo: bar\r\n\r\n')