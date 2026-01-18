import base64
from enum import Enum, IntEnum
from hyperframe.exceptions import InvalidPaddingError
from hyperframe.frame import (
from hpack.hpack import Encoder, Decoder
from hpack.exceptions import HPACKError, OversizedHeaderListError
from .config import H2Configuration
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .frame_buffer import FrameBuffer
from .settings import Settings, SettingCodes
from .stream import H2Stream, StreamClosedBy
from .utilities import SizeLimitDict, guard_increment_window
from .windows import WindowManager
def remote_flow_control_window(self, stream_id):
    """
        Returns the maximum amount of data the remote peer can send on stream
        ``stream_id``.

        This value will never be larger than the total data that can be sent on
        the connection: even if the given stream allows more data, the
        connection window provides a logical maximum to the amount of data that
        can be sent.

        The maximum data that can be sent in a single data frame on a stream
        is either this value, or the maximum frame size, whichever is
        *smaller*.

        :param stream_id: The ID of the stream whose flow control window is
            being queried.
        :type stream_id: ``int``
        :returns: The amount of data in bytes that can be received on the
            stream before the flow control window is exhausted.
        :rtype: ``int``
        """
    stream = self._get_stream_by_id(stream_id)
    return min(self.inbound_flow_control_window, stream.inbound_flow_control_window)