from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def send_push_on_closed_stream(self, previous_state):
    """
        Called when an attempt is made to push on an already-closed stream.

        This essentially overrides the standard logic by providing a more
        useful error message. It's necessary because simply indicating that the
        stream is closed is not enough: there is now a new stream that is not
        allowed to be there. The only recourse is to tear the whole connection
        down.
        """
    raise ProtocolError('Attempted to push on closed stream.')