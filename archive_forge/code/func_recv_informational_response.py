from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def recv_informational_response(self, previous_state):
    """
        Called when an informational header block is received (that is, a block
        where the :status header has a 1XX value).
        """
    if self.headers_received:
        raise ProtocolError('Informational response after final response')
    event = InformationalResponseReceived()
    event.stream_id = self.stream_id
    return [event]