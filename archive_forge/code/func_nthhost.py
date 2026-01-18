from __future__ import absolute_import, division, print_function
from functools import partial
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddr_utils import (
def nthhost(value, query=''):
    """Returns the nth host within a network described by value."""
    try:
        vtype = ipaddr(value, 'type')
        if vtype == 'address':
            v = ipaddr(value, 'cidr')
        elif vtype == 'network':
            v = ipaddr(value, 'subnet')
        value = netaddr.IPNetwork(v)
    except Exception:
        return False
    if not query:
        return False
    try:
        nth = int(query)
        if value.size > nth:
            return str(value[nth])
    except ValueError:
        return False
    return False