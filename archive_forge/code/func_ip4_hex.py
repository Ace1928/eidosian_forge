from __future__ import absolute_import, division, print_function
from functools import partial
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddr_utils import _need_netaddr
def ip4_hex(arg, delimiter=''):
    """Convert an IPv4 address to Hexadecimal notation"""
    try:
        ip = netaddr.IPAddress(arg)
    except (netaddr.AddrFormatError, ValueError):
        msg = 'You must pass a valid IP address; {0} is invalid'.format(arg)
        raise AnsibleFilterError(msg)
    numbers = list(map(int, arg.split('.')))
    return '{0:02x}{sep}{1:02x}{sep}{2:02x}{sep}{3:02x}'.format(*numbers, sep=delimiter)