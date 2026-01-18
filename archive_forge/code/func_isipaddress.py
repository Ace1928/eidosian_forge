from __future__ import absolute_import, division, print_function
from functools import total_ordering
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.network import is_masklen, to_netmask
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
def isipaddress(data):
    """
    Checks if the passed string is
    a valid IPv4 or IPv6 address
    """
    if not HAS_IPADDRESS:
        raise Exception(missing_required_lib('ipaddress'))
    isipaddress = True
    try:
        ipaddress.ip_address(data)
    except ValueError:
        isipaddress = False
    return isipaddress