from __future__ import absolute_import
from io import StringIO
import struct
import time
import dns.edns
import dns.exception
import dns.flags
import dns.name
import dns.opcode
import dns.entropy
import dns.rcode
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.rrset
import dns.renderer
import dns.tsig
import dns.wiredata
from ._compat import long, xrange, string_types
def is_response(self, other):
    """Is this message a response to *other*?

        Returns a ``bool``.
        """
    if other.flags & dns.flags.QR == 0 or self.id != other.id or dns.opcode.from_flags(self.flags) != dns.opcode.from_flags(other.flags):
        return False
    if dns.rcode.from_flags(other.flags, other.ednsflags) != dns.rcode.NOERROR:
        return True
    if dns.opcode.is_update(self.flags):
        return True
    for n in self.question:
        if n not in other.question:
            return False
    for n in other.question:
        if n not in self.question:
            return False
    return True