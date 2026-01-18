import re
from hashlib import md5
from typing import List
from twisted.internet import defer, error, interfaces
from twisted.mail._except import (
from twisted.protocols import basic, policies
from twisted.python import log
def state_LONG(self, line):
    """
        Handle server responses for the LONG state in which the server is
        expected to send a non-initial line of a multi-line response.

        On receipt of the last line of the response, clean up, fire the
        deferred which is waiting on receipt of a complete response, and
        transition the state to WAITING. Otherwise, pass the line to the
        transform function, if provided, and then the consumer function.

        @type line: L{bytes}
        @param line: A line received from the server.

        @rtype: L{bytes}
        @return: The next state.
        """
    if line == b'.':
        consumer = self._consumer
        deferred = self._waiting
        self._consumer = self._waiting = self._xform = None
        self._unblock()
        deferred.callback(consumer)
        return 'WAITING'
    else:
        if self._xform is not None:
            self._consumer(self._xform(line))
        else:
            self._consumer(line)
        return 'LONG'