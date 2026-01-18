import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def sendClose(self, channel):
    """
        Close a channel.

        @type channel:  subclass of L{SSHChannel}
        """
    if channel.localClosed:
        return
    self._log.info('sending close {id}', id=channel.id)
    self.transport.sendPacket(MSG_CHANNEL_CLOSE, struct.pack('>L', self.channelsToRemoteChannel[channel]))
    channel.localClosed = True
    if channel.localClosed and channel.remoteClosed:
        self.channelClosed(channel)