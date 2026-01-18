import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def ssh_CHANNEL_OPEN_FAILURE(self, packet):
    """
        The other side did not accept our MSG_CHANNEL_OPEN request.  Payload::
            uint32  local channel number
            uint32  reason code
            string  reason description

        Find the channel using the local channel number and notify it by
        calling its openFailed() method.
        """
    localChannel, reasonCode = struct.unpack('>2L', packet[:8])
    reasonDesc = common.getNS(packet[8:])[0]
    channel = self.channels[localChannel]
    del self.channels[localChannel]
    channel.conn = self
    reason = error.ConchError(reasonDesc, reasonCode)
    channel.openFailed(reason)