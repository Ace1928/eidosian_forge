from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def send_reset_stream(self, previous_state):
    """
        Called when an attempt is made to send RST_STREAM in a non-closed
        stream state.
        """
    self.stream_closed_by = StreamClosedBy.SEND_RST_STREAM