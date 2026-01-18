import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def iter_hosts(self):
    """
        A generator that provides all the IP addresses that can be assigned
        to hosts within the range of this IP object's subnet.

        - for IPv4, the network and broadcast addresses are excluded, excepted           when using /31 or /32 subnets as per RFC 3021.

        - for IPv6, only Subnet-Router anycast address (first address in the           network) is excluded as per RFC 4291 section 2.6.1, excepted when using           /127 or /128 subnets as per RFC 6164.

        :return: an IPAddress iterator
        """
    first_usable_address, last_usable_address = self._usable_range()
    return iter_iprange(IPAddress(first_usable_address, self._module.version), IPAddress(last_usable_address, self._module.version))