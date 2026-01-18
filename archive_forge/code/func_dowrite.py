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
def dowrite(writer: Callable[..., None], obj: Any) -> bytes:
    got_list: List[bytes] = []
    writer(obj, got_list.append)
    return b''.join(got_list)