from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def recv_push_on_closed_stream(self, previous_state):
    """
        Called when a PUSH_PROMISE frame is received on a full stop
        stream.

        If the stream was closed by us sending a RST_STREAM frame, then we
        presume that the PUSH_PROMISE was in flight when we reset the parent
        stream. Rathen than accept the new stream, we just reset it.
        Otherwise, we should call this a PROTOCOL_ERROR: pushing a stream on a
        naturally closed stream is a real problem because it creates a brand
        new stream that the remote peer now believes exists.
        """
    assert self.stream_closed_by is not None
    if self.stream_closed_by == StreamClosedBy.SEND_RST_STREAM:
        raise StreamClosedError(self.stream_id)
    else:
        raise ProtocolError('Attempted to push on closed stream.')