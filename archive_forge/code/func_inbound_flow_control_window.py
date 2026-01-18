from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
@property
def inbound_flow_control_window(self):
    """
        The size of the inbound flow control window for the stream. This is
        rarely publicly useful: instead, use :meth:`remote_flow_control_window
        <h2.stream.H2Stream.remote_flow_control_window>`. This shortcut is
        largely present to provide a shortcut to this data.
        """
    return self._inbound_window_manager.current_window_size