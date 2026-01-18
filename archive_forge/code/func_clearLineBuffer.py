import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
def clearLineBuffer(self):
    """
        Clear buffered data.

        @return: All of the cleared buffered data.
        @rtype: C{bytes}
        """
    b, self._buffer = (self._buffer, b'')
    return b