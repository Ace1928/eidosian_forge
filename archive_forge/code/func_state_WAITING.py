import re
from hashlib import md5
from typing import List
from twisted.internet import defer, error, interfaces
from twisted.mail._except import (
from twisted.protocols import basic, policies
from twisted.python import log
def state_WAITING(self, line):
    """
        Log an error for server responses received in the WAITING state during
        which the server is not expected to send anything.

        @type line: L{bytes}
        @param line: A line received from the server.
        """
    log.msg('Illegal line from server: ' + repr(line))