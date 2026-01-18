from __future__ import absolute_import, division, print_function
from functools import partial
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddr_utils import (
def previous_nth_usable(value, offset):
    """
    Returns the previous nth usable ip within a network described by value.
    """
    try:
        vtype = ipaddr(value, 'type')
        if vtype == 'address':
            v = ipaddr(value, 'cidr')
        elif vtype == 'network':
            v = ipaddr(value, 'subnet')
        v = netaddr.IPNetwork(v)
    except Exception:
        return False
    if not isinstance(offset, int):
        raise AnsibleFilterError('Must pass in an integer')
    if v.size > 1:
        first_usable, last_usable = _first_last(v)
        nth_ip = int(netaddr.IPAddress(int(v.ip) - offset))
        if nth_ip >= first_usable and nth_ip <= last_usable:
            return str(netaddr.IPAddress(int(v.ip) - offset))