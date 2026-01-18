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
def initiate_upgrade_connection(self, settings_header=None):
    """
        Call to initialise the connection object for use with an upgraded
        HTTP/2 connection (i.e. a connection negotiated using the
        ``Upgrade: h2c`` HTTP header).

        This method differs from :meth:`initiate_connection
        <h2.connection.H2Connection.initiate_connection>` in several ways.
        Firstly, it handles the additional SETTINGS frame that is sent in the
        ``HTTP2-Settings`` header field. When called on a client connection,
        this method will return a bytestring that the caller can put in the
        ``HTTP2-Settings`` field they send on their initial request. When
        called on a server connection, the user **must** provide the value they
        received from the client in the ``HTTP2-Settings`` header field to the
        ``settings_header`` argument, which will be used appropriately.

        Additionally, this method sets up stream 1 in a half-closed state
        appropriate for this side of the connection, to reflect the fact that
        the request is already complete.

        Finally, this method also prepares the appropriate preamble to be sent
        after the upgrade.

        .. versionadded:: 2.3.0

        :param settings_header: (optional, server-only): The value of the
             ``HTTP2-Settings`` header field received from the client.
        :type settings_header: ``bytes``

        :returns: For clients, a bytestring to put in the ``HTTP2-Settings``.
            For servers, returns nothing.
        :rtype: ``bytes`` or ``None``
        """
    self.config.logger.debug('Upgrade connection. Current settings: %s', self.local_settings)
    frame_data = None
    self.initiate_connection()
    if self.config.client_side:
        f = SettingsFrame(0)
        for setting, value in self.local_settings.items():
            f.settings[setting] = value
        frame_data = f.serialize_body()
        frame_data = base64.urlsafe_b64encode(frame_data)
    elif settings_header:
        settings_header = base64.urlsafe_b64decode(settings_header)
        f = SettingsFrame(0)
        f.parse_body(settings_header)
        self._receive_settings_frame(f)
    connection_input = ConnectionInputs.SEND_HEADERS if self.config.client_side else ConnectionInputs.RECV_HEADERS
    self.config.logger.debug('Process input %s', connection_input)
    self.state_machine.process_input(connection_input)
    self._begin_new_stream(stream_id=1, allowed_ids=AllowedStreamIDs.ODD)
    self.streams[1].upgrade(self.config.client_side)
    return frame_data