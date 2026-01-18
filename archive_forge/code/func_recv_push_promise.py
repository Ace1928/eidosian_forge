from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def recv_push_promise(self, previous_state):
    """
        Fires on the already-existing stream when a PUSH_PROMISE frame is
        received. We may only receive PUSH_PROMISE frames if we're a client.

        Fires a PushedStreamReceived event.
        """
    if not self.client:
        if self.client is None:
            msg = 'Idle streams cannot receive pushes'
        else:
            msg = 'Cannot receive pushed streams as a server'
        raise ProtocolError(msg)
    event = PushedStreamReceived()
    event.parent_stream_id = self.stream_id
    return [event]