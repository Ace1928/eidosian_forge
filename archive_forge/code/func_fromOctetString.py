import math
import sys
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import integer
from pyasn1.compat import octets
from pyasn1.type import base
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import tagmap
@classmethod
def fromOctetString(cls, value, internalFormat=False, prepend=None, padding=0):
    """Create a |ASN.1| object initialized from a string.

        Parameters
        ----------
        value: :class:`str` (Py2) or :class:`bytes` (Py3)
            Text string like '\\\\x01\\\\xff' (Py2) or b'\\\\x01\\\\xff' (Py3)
        """
    value = SizedInteger(integer.from_bytes(value) >> padding).setBitLength(len(value) * 8 - padding)
    if prepend is not None:
        value = SizedInteger(SizedInteger(prepend) << len(value) | value).setBitLength(len(prepend) + len(value))
    if not internalFormat:
        value = cls(value)
    return value