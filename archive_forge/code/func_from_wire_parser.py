import struct
import dns.immutable
import dns.rdata
@classmethod
def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
    preference = parser.get_uint16()
    locator32 = parser.get_remaining()
    return cls(rdclass, rdtype, preference, locator32)