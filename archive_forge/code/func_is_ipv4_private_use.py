import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def is_ipv4_private_use(self):
    """
        Returns ``True`` if this address is an IPv4 private-use address as defined in
        :rfc:`1918`.

        The private-use address blocks:

        * ``10.0.0.0/8``
        * ``172.16.0.0/12``
        * ``192.168.0.0/16``

        .. note:: |ipv4_in_ipv6_handling|

        .. versionadded:: 0.10.0
        """
    return self._module.version == 4 and any((self in cidr for cidr in IPV4_PRIVATE_USE))