from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
def netmask_length_to_netmask(ip_address, length, module):
    """
    input: ip_address and netmask length
    output: netmask in dot notation
    """
    return str(_get_ipv4orv6_network(ip_address, length, False, module).netmask)