import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def gotGlobalRequest(self, requestType, data):
    """
        We got a global request.  pretty much, this is just used by the client
        to request that we forward a port from the server to the client.
        Returns either:
            - 1: request accepted
            - 1, <data>: request accepted with request specific data
            - 0: request denied

        By default, this dispatches to a method 'global_requestType' with
        -'s in requestType replaced with _'s.  The found method is passed data.
        If this method cannot be found, this method returns 0.  Otherwise, it
        returns the return value of that method.

        @type requestType:  L{bytes}
        @type data:         L{bytes}
        @rtype:             L{int}/L{tuple}
        """
    self._log.debug('got global {requestType} request', requestType=requestType)
    if hasattr(self.transport, 'avatar'):
        return self.transport.avatar.gotGlobalRequest(requestType, data)
    requestType = nativeString(requestType.replace(b'-', b'_'))
    f = getattr(self, 'global_%s' % requestType, None)
    if not f:
        return 0
    return f(data)