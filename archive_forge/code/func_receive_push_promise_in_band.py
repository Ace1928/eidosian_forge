from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def receive_push_promise_in_band(self, promised_stream_id, headers, header_encoding):
    """
        Receives a push promise frame sent on this stream, pushing a remote
        stream. This is called on the stream that has the PUSH_PROMISE sent
        on it.
        """
    self.config.logger.debug('Receive Push Promise on %r for remote stream %d', self, promised_stream_id)
    events = self.state_machine.process_input(StreamInputs.RECV_PUSH_PROMISE)
    events[0].pushed_stream_id = promised_stream_id
    hdr_validation_flags = self._build_hdr_validation_flags(events)
    events[0].headers = self._process_received_headers(headers, hdr_validation_flags, header_encoding)
    return ([], events)