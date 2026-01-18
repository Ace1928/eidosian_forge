import socket
from typing import Any, Optional, Tuple
import dns.ipv4
import dns.ipv6
def any_for_af(af):
    """Return the 'any' address for the specified address family."""
    if af == socket.AF_INET:
        return '0.0.0.0'
    elif af == socket.AF_INET6:
        return '::'
    raise NotImplementedError(f'unknown address family {af}')