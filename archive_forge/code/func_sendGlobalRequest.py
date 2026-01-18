import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def sendGlobalRequest(self, request, data, wantReply=0):
    """
        Send a global request for this connection.  Current this is only used
        for remote->local TCP forwarding.

        @type request:      L{bytes}
        @type data:         L{bytes}
        @type wantReply:    L{bool}
        @rtype:             C{Deferred}/L{None}
        """
    self.transport.sendPacket(MSG_GLOBAL_REQUEST, common.NS(request) + (wantReply and b'\xff' or b'\x00') + data)
    if wantReply:
        d = defer.Deferred()
        self.deferreds['global'].append(d)
        return d