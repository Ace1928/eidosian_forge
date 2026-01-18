import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def sendEOF(self, channel):
    """
        Send an EOF (End of File) for a channel.

        @type channel:  subclass of L{SSHChannel}
        """
    if channel.localClosed:
        return
    self._log.debug('sending eof')
    self.transport.sendPacket(MSG_CHANNEL_EOF, struct.pack('>L', self.channelsToRemoteChannel[channel]))