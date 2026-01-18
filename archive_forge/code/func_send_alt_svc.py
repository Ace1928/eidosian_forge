from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def send_alt_svc(self, previous_state):
    """
        Called when sending an ALTSVC frame on this stream.

        For consistency with the restrictions we apply on receiving ALTSVC
        frames in ``recv_alt_svc``, we want to restrict when users can send
        ALTSVC frames to the situations when we ourselves would accept them.

        That means: when we are a server, when we have received the request
        headers, and when we have not yet sent our own response headers.
        """
    if self.headers_sent:
        raise ProtocolError('Cannot send ALTSVC after sending response headers.')
    return