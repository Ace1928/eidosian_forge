from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def recv_alt_svc(self, previous_state):
    """
        Called when receiving an ALTSVC frame.

        RFC 7838 allows us to receive ALTSVC frames at any stream state, which
        is really absurdly overzealous. For that reason, we want to limit the
        states in which we can actually receive it. It's really only sensible
        to receive it after we've sent our own headers and before the server
        has sent its header block: the server can't guarantee that we have any
        state around after it completes its header block, and the server
        doesn't know what origin we're talking about before we've sent ours.

        For that reason, this function applies a few extra checks on both state
        and some of the little state variables we keep around. If those suggest
        an unreasonable situation for the ALTSVC frame to have been sent in,
        we quietly ignore it (as RFC 7838 suggests).

        This function is also *not* always called by the state machine. In some
        states (IDLE, RESERVED_LOCAL, CLOSED) we don't bother to call it,
        because we know the frame cannot be valid in that state (IDLE because
        the server cannot know what origin the stream applies to, CLOSED
        because the server cannot assume we still have state around,
        RESERVED_LOCAL because by definition if we're in the RESERVED_LOCAL
        state then *we* are the server).
        """
    if self.client is False:
        return []
    if self.headers_received:
        return []
    return [AlternativeServiceAvailable()]