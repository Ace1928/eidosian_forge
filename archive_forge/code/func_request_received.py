from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def request_received(self, previous_state):
    """
        Fires when a request is received.
        """
    assert not self.headers_received
    assert not self.trailers_received
    self.client = False
    self.headers_received = True
    event = RequestReceived()
    event.stream_id = self.stream_id
    return [event]