from socket import AF_INET, AF_INET6, inet_pton
from typing import Iterable, List, Optional
from zope.interface import implementer
from twisted.internet import interfaces, main
from twisted.python import failure, reflect
from twisted.python.compat import lazyByteSlice
def isIPAddress(addr: str, family: int=AF_INET) -> bool:
    """
    Determine whether the given string represents an IP address of the given
    family; by default, an IPv4 address.

    @param addr: A string which may or may not be the decimal dotted
        representation of an IPv4 address.
    @param family: The address family to test for; one of the C{AF_*} constants
        from the L{socket} module.  (This parameter has only been available
        since Twisted 17.1.0; previously L{isIPAddress} could only test for IPv4
        addresses.)

    @return: C{True} if C{addr} represents an IPv4 address, C{False} otherwise.
    """
    if isinstance(addr, bytes):
        try:
            addr = addr.decode('ascii')
        except UnicodeDecodeError:
            return False
    if family == AF_INET6:
        addr = addr.split('%', 1)[0]
    elif family == AF_INET:
        if addr.count('.') != 3:
            return False
    else:
        raise ValueError(f'unknown address family {family!r}')
    try:
        inet_pton(family, addr)
    except (ValueError, OSError):
        return False
    return True