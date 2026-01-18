from io import BytesIO
import struct
import sys
import copy
import encodings.idna
import dns.exception
import dns.wiredata
from ._compat import long, binary_type, text_type, unichr, maybe_decode
def is_superdomain(self, other):
    """Is self a superdomain of other?

        Note that the notion of superdomain includes equality, e.g.
        "dnpython.org" is a superdomain of itself.

        Returns a ``bool``.
        """
    nr, o, nl = self.fullcompare(other)
    if nr == NAMERELN_SUPERDOMAIN or nr == NAMERELN_EQUAL:
        return True
    return False