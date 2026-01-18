from io import BytesIO
import struct
import sys
import copy
import encodings.idna
import dns.exception
import dns.wiredata
from ._compat import long, binary_type, text_type, unichr, maybe_decode
def derelativize(self, origin):
    """If the name is a relative name, return a new name which is the
        concatenation of the name and origin.  Otherwise return the name.

        For example, derelativizing ``www`` to origin ``dnspython.org.``
        returns the name ``www.dnspython.org.``.  Derelativizing ``example.``
        to origin ``dnspython.org.`` returns ``example.``.

        Returns a ``dns.name.Name``.
        """
    if not self.is_absolute():
        return self.concatenate(origin)
    else:
        return self