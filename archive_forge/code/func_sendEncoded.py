import copy
import struct
from io import BytesIO
from twisted.internet import protocol
from twisted.persisted import styles
from twisted.python import log
from twisted.python.compat import iterbytes
from twisted.python.reflect import fullyQualifiedName
def sendEncoded(self, obj):
    """
        Send the encoded representation of the given object:

        @param obj: An object to encode and send.

        @raise BananaError: If the given object is not an instance of one of
            the types supported by Banana.

        @return: L{None}
        """
    encodeStream = BytesIO()
    self._encode(obj, encodeStream.write)
    value = encodeStream.getvalue()
    self.transport.write(value)