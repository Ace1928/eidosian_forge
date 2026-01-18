import socket
from typing import Any, Optional, Tuple
import dns.ipv4
import dns.ipv6
def low_level_address_tuple(high_tuple: Tuple[str, int], af: Optional[int]=None) -> Any:
    """Given a "high-level" address tuple, i.e.
    an (address, port) return the appropriate "low-level" address tuple
    suitable for use in socket calls.

    If an *af* other than ``None`` is provided, it is assumed the
    address in the high-level tuple is valid and has that af.  If af
    is ``None``, then af_for_address will be called.
    """
    address, port = high_tuple
    if af is None:
        af = af_for_address(address)
    if af == AF_INET:
        return (address, port)
    elif af == AF_INET6:
        i = address.find('%')
        if i < 0:
            return (address, port, 0, 0)
        addrpart = address[:i]
        scope = address[i + 1:]
        if scope.isdigit():
            return (addrpart, port, 0, int(scope))
        try:
            return (addrpart, port, 0, socket.if_nametoindex(scope))
        except AttributeError:
            ai_flags = socket.AI_NUMERICHOST
            (*_, tup), *_ = socket.getaddrinfo(address, port, flags=ai_flags)
            return tup
    else:
        raise NotImplementedError(f'unknown address family {af}')