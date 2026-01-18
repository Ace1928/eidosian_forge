from socket import AF_INET, AF_INET6, inet_pton
from typing import Iterable, List, Optional
from zope.interface import implementer
from twisted.internet import interfaces, main
from twisted.python import failure, reflect
from twisted.python.compat import lazyByteSlice
def isIPv6Address(addr: str) -> bool:
    """
    Determine whether the given string represents an IPv6 address.

    @param addr: A string which may or may not be the hex
        representation of an IPv6 address.
    @type addr: C{str}

    @return: C{True} if C{addr} represents an IPv6 address, C{False}
        otherwise.
    @rtype: C{bool}
    """
    return isIPAddress(addr, AF_INET6)