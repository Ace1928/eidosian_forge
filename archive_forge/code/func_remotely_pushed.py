from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def remotely_pushed(self, pushed_headers):
    """
        Mark this stream as one that was pushed by the remote peer. Must be
        called immediately after initialization. Sends no frames, simply
        updates the state machine.
        """
    self.config.logger.debug('%r pushed by remote peer', self)
    events = self.state_machine.process_input(StreamInputs.RECV_PUSH_PROMISE)
    self._authority = authority_from_headers(pushed_headers)
    return ([], events)