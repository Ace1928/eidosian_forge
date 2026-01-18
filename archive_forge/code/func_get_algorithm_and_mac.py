import hashlib
import hmac
import struct
import dns.exception
import dns.rdataclass
import dns.name
from ._compat import long, string_types, text_type
def get_algorithm_and_mac(wire, tsig_rdata, tsig_rdlen):
    """Return the tsig algorithm for the specified tsig_rdata
    @raises FormError: The TSIG is badly formed.
    """
    current = tsig_rdata
    aname, used = dns.name.from_wire(wire, current)
    current = current + used
    upper_time, lower_time, fudge, mac_size = struct.unpack('!HIHH', wire[current:current + 10])
    current += 10
    mac = wire[current:current + mac_size]
    current += mac_size
    if current > tsig_rdata + tsig_rdlen:
        raise dns.exception.FormError
    return (aname, mac)