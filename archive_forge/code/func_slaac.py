from __future__ import absolute_import, division, print_function
from functools import partial
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddr_utils import (
def slaac(value, query=''):
    """Get the SLAAC address within given network"""
    try:
        vtype = ipaddr(value, 'type')
        if vtype == 'address':
            v = ipaddr(value, 'cidr')
        elif vtype == 'network':
            v = ipaddr(value, 'subnet')
        if ipaddr(value, 'version') != 6:
            return False
        value = netaddr.IPNetwork(v)
    except Exception:
        return False
    if not query:
        return False
    try:
        mac = hwaddr(query, alias='slaac')
        eui = netaddr.EUI(mac)
    except Exception:
        return False
    return str(eui.ipv6(value.network))