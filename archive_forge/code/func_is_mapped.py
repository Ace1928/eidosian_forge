import re
import binascii
import dns.exception
import dns.ipv4
from ._compat import xrange, binary_type, maybe_decode
def is_mapped(address):
    """Is the specified address a mapped IPv4 address?

    *address*, a ``binary`` is an IPv6 address in binary form.

    Returns a ``bool``.
    """
    return address.startswith(_mapped_prefix)