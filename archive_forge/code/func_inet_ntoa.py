import struct
import dns.exception
from ._compat import binary_type
def inet_ntoa(address):
    """Convert an IPv4 address in binary form to text form.

    *address*, a ``binary``, the IPv4 address in binary form.

    Returns a ``text``.
    """
    if len(address) != 4:
        raise dns.exception.SyntaxError
    if not isinstance(address, bytearray):
        address = bytearray(address)
    return '%u.%u.%u.%u' % (address[0], address[1], address[2], address[3])