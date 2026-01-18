import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def is_unicast(self):
    """:return: ``True`` if this IP is unicast, ``False`` otherwise"""
    return not self.is_multicast()