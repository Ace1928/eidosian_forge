from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def send_informational_response(self, previous_state):
    """
        Called when an informational header block is sent (that is, a block
        where the :status header has a 1XX value).

        Only enforces that these are sent *before* final headers are sent.
        """
    if self.headers_sent:
        raise ProtocolError('Information response after final response')
    event = _ResponseSent()
    return [event]