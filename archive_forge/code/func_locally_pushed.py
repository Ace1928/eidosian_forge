from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def locally_pushed(self):
    """
        Mark this stream as one that was pushed by this peer. Must be called
        immediately after initialization. Sends no frames, simply updates the
        state machine.
        """
    events = self.state_machine.process_input(StreamInputs.SEND_PUSH_PROMISE)
    assert not events
    return []