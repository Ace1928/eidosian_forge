import copy
import struct
from io import BytesIO
from twisted.internet import protocol
from twisted.persisted import styles
from twisted.python import log
from twisted.python.compat import iterbytes
from twisted.python.reflect import fullyQualifiedName
def setPrefixLimit(self, limit):
    """
        Set the prefix limit for decoding done by this protocol instance.

        @see: L{setPrefixLimit}
        """
    self.prefixLimit = limit
    self._smallestLongInt = -2 ** (limit * 7) + 1
    self._smallestInt = -2 ** 31
    self._largestInt = 2 ** 31 - 1
    self._largestLongInt = 2 ** (limit * 7) - 1