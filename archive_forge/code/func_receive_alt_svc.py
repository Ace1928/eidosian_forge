from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def receive_alt_svc(self, frame):
    """
        An Alternative Service frame was received on the stream. This frame
        inherits the origin associated with this stream.
        """
    self.config.logger.debug('Receive Alternative Service frame on stream %r', self)
    if frame.origin:
        return ([], [])
    events = self.state_machine.process_input(StreamInputs.RECV_ALTERNATIVE_SERVICE)
    if events:
        assert isinstance(events[0], AlternativeServiceAvailable)
        events[0].origin = self._authority
        events[0].field_value = frame.field
    return ([], events)