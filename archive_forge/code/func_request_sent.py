from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def request_sent(self, previous_state):
    """
        Fires when a request is sent.
        """
    self.client = True
    self.headers_sent = True
    event = _RequestSent()
    return [event]