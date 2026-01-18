from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def reset_stream_on_error(self, previous_state):
    """
        Called when we need to forcefully emit another RST_STREAM frame on
        behalf of the state machine.

        If this is the first time we've done this, we should also hang an event
        off the StreamClosedError so that the user can be informed. We know
        it's the first time we've done this if the stream is currently in a
        state other than CLOSED.
        """
    self.stream_closed_by = StreamClosedBy.SEND_RST_STREAM
    error = StreamClosedError(self.stream_id)
    event = StreamReset()
    event.stream_id = self.stream_id
    event.error_code = ErrorCodes.STREAM_CLOSED
    event.remote_reset = False
    error._events = [event]
    raise error