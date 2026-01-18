import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def ssh_CHANNEL_SUCCESS(self, packet):
    """
        Our channel request to the other side succeeded.  Payload::
            uint32  local channel number

        Get the C{Deferred} out of self.deferreds and call it back.
        """
    localChannel = struct.unpack('>L', packet[:4])[0]
    if self.deferreds.get(localChannel):
        d = self.deferreds[localChannel].pop(0)
        d.callback('')