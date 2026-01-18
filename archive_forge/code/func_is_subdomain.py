from io import BytesIO
import struct
import sys
import copy
import encodings.idna
import dns.exception
import dns.wiredata
from ._compat import long, binary_type, text_type, unichr, maybe_decode
def is_subdomain(self, other):
    """Is self a subdomain of other?

        Note that the notion of subdomain includes equality, e.g.
        "dnpython.org" is a subdomain of itself.

        Returns a ``bool``.
        """
    nr, o, nl = self.fullcompare(other)
    if nr == NAMERELN_SUBDOMAIN or nr == NAMERELN_EQUAL:
        return True
    return False