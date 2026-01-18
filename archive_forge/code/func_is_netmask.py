import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def is_netmask(self):
    """
        :return: ``True`` if this IP address network mask, ``False`` otherwise.
        """
    int_val = (self._value ^ self._module.max_int) + 1
    return int_val & int_val - 1 == 0