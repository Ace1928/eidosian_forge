from __future__ import absolute_import, division, print_function
from functools import partial
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddr_utils import _need_netaddr
def ipcut(value, amount):
    ipv6_oct = []
    try:
        ip = netaddr.IPAddress(value)
        ipv6address = ip.bits().replace(':', '')
    except (netaddr.AddrFormatError, ValueError):
        msg = 'You must pass a valid IP address; {0} is invalid'.format(value)
        raise AnsibleFilterError(msg)
    if not isinstance(amount, int):
        msg = 'You must pass an integer for arithmetic; {0} is not a valid integer'.format(amount)
        raise AnsibleFilterError(msg)
    elif amount < 0:
        ipsub = ipv6address[amount:]
    else:
        ipsub = ipv6address[0:amount]
    ipsubfinal = []
    for i in range(0, len(ipsub), 16):
        oct_sub = i + 16
        ipsubfinal.append(ipsub[i:oct_sub])
    for i in ipsubfinal:
        x = hex(int(i, 2))
        ipv6_oct.append(x.replace('0x', ''))
    return str(':'.join(ipv6_oct))