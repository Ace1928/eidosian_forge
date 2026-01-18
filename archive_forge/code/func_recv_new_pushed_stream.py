from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def recv_new_pushed_stream(self, previous_state):
    """
        Fires on the newly pushed stream, when pushed by the remote peer.

        No event here, but definitionally this peer must be a client.
        """
    assert self.client is None
    self.client = True
    self.headers_sent = True
    return []