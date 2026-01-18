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
def test__obsolete_line_fold_bytes() -> None:
    assert list(_obsolete_line_fold([b'aaa', b'bbb', b'  ccc', b'ddd'])) == [b'aaa', bytearray(b'bbb ccc'), b'ddd']