import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def ssh_REQUEST_SUCCESS(self, packet):
    """
        Our global request succeeded.  Get the appropriate Deferred and call
        it back with the packet we received.
        """
    self._log.debug('global request success')
    self.deferreds['global'].pop(0).callback(packet)