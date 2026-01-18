import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def classful_prefix(octet):
    octet = int(octet)
    if not 0 <= octet <= 255:
        raise IndexError('Invalid octet: %r!' % octet)
    if 0 <= octet <= 127:
        return 8
    elif 128 <= octet <= 191:
        return 16
    elif 192 <= octet <= 223:
        return 24
    elif 224 <= octet <= 239:
        return 4
    return 32