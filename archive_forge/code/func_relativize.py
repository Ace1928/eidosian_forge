from io import BytesIO
import struct
import sys
import copy
import encodings.idna
import dns.exception
import dns.wiredata
from ._compat import long, binary_type, text_type, unichr, maybe_decode
def relativize(self, origin):
    """If the name is a subdomain of *origin*, return a new name which is
        the name relative to origin.  Otherwise return the name.

        For example, relativizing ``www.dnspython.org.`` to origin
        ``dnspython.org.`` returns the name ``www``.  Relativizing ``example.``
        to origin ``dnspython.org.`` returns ``example.``.

        Returns a ``dns.name.Name``.
        """
    if origin is not None and self.is_subdomain(origin):
        return Name(self[:-len(origin)])
    else:
        return self