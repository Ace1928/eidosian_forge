import struct
import dns.exception
import dns.rdata
import dns.name
@classmethod
def from_wire(cls, rdclass, rdtype, wire, current, rdlen, origin=None):
    mname, cused = dns.name.from_wire(wire[:current + rdlen], current)
    current += cused
    rdlen -= cused
    rname, cused = dns.name.from_wire(wire[:current + rdlen], current)
    current += cused
    rdlen -= cused
    if rdlen != 20:
        raise dns.exception.FormError
    five_ints = struct.unpack('!IIIII', wire[current:current + rdlen])
    if origin is not None:
        mname = mname.relativize(origin)
        rname = rname.relativize(origin)
    return cls(rdclass, rdtype, mname, rname, five_ints[0], five_ints[1], five_ints[2], five_ints[3], five_ints[4])