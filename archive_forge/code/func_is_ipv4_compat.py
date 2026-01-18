import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def is_ipv4_compat(self):
    """
        :return: ``True`` if this IP is IPv4-mapped IPv6 address, ``False``
            otherwise.
        """
    return self._module.version == 6 and self._value >> 32 == 0