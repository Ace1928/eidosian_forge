import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def ssh_CHANNEL_CLOSE(self, packet):
    """
        The other side is closing its end; it does not want to receive any
        more data.  Payload::
            uint32  local channel number

        Notify the channnel by calling its closeReceived() method.  If
        the channel has also sent a close message, call self.channelClosed().
        """
    localChannel = struct.unpack('>L', packet[:4])[0]
    channel = self.channels[localChannel]
    channel.closeReceived()
    channel.remoteClosed = True
    if channel.localClosed and channel.remoteClosed:
        self.channelClosed(channel)