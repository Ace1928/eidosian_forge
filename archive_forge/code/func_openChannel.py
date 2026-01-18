import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def openChannel(self, channel, extra=b''):
    """
        Open a new channel on this connection.

        @type channel:  subclass of C{SSHChannel}
        @type extra:    L{bytes}
        """
    self._log.info('opening channel {id} with {localWindowSize} {localMaxPacket}', id=self.localChannelID, localWindowSize=channel.localWindowSize, localMaxPacket=channel.localMaxPacket)
    self.transport.sendPacket(MSG_CHANNEL_OPEN, common.NS(channel.name) + struct.pack('>3L', self.localChannelID, channel.localWindowSize, channel.localMaxPacket) + extra)
    channel.id = self.localChannelID
    self.channels[self.localChannelID] = channel
    self.localChannelID += 1