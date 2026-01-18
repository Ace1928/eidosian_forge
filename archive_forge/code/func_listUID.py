import re
from hashlib import md5
from typing import List
from twisted.internet import defer, error, interfaces
from twisted.mail._except import (
from twisted.protocols import basic, policies
from twisted.python import log
def listUID(self, consumer=None):
    """
        Send a UIDL command to retrieve the UIDs of all messages on the server.

        @type consumer: L{None} or callable that takes
            2-L{tuple} of (0) L{int}, (1) L{bytes}
        @param consumer: A function which consumes the 0-based message index
            and UID derived from the server response.

        @rtype: L{Deferred <defer.Deferred>} which fires with L{list} of
            L{object} or callable that takes 2-L{tuple} of (0) L{int},
            (1) L{bytes}
        @return: A deferred which fires when the entire response has been
            received.  When a consumer is not provided, the return value is a
            list of message sizes.  Otherwise, it returns the consumer itself.
        """
    return self._consumeOrSetItem(b'UIDL', None, consumer, _uidXform)