from __future__ import absolute_import, division, print_function
from functools import total_ordering
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.network import is_masklen, to_netmask
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
def prefix_to_address_wildcard(prefix):
    """Converts a IPv4 prefix into address and
        wildcard mask

    :returns: IPv4 address and wildcard mask
    """
    if not HAS_IPADDRESS:
        raise Exception(missing_required_lib('ipaddress'))
    wildcard = []
    subnet = to_text(ipaddress.IPv4Network(to_text(prefix)).netmask)
    for x in subnet.split('.'):
        component = 255 - int(x)
        wildcard.append(str(component))
    wildcard = '.'.join(wildcard)
    return (prefix.split('/')[0], wildcard)