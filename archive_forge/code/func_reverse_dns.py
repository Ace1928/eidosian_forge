import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
@property
def reverse_dns(self):
    """The reverse DNS lookup record for this IP address"""
    return self._module.int_to_arpa(self._value)