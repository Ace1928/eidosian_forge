from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def patch_want_with_default(self, want, ac_address_only=False):
    new_want = {}
    if want is None:
        if ac_address_only:
            new_want = {'anycast_address': {'ipv4': True, 'ipv6': True, 'mac_address': None}}
        else:
            new_want = {'hostname': 'sonic', 'interface_naming': 'native', 'anycast_address': {'ipv4': True, 'ipv6': True, 'mac_address': None}}
    else:
        new_want = want.copy()
        new_anycast = {}
        anycast = want.get('anycast_address', None)
        if not anycast:
            new_anycast = {'ipv4': True, 'ipv6': True, 'mac_address': None}
        else:
            new_anycast = anycast.copy()
            ipv4 = anycast.get('ipv4', None)
            if ipv4 is None:
                new_anycast['ipv4'] = True
            ipv6 = anycast.get('ipv6', None)
            if ipv6 is None:
                new_anycast['ipv6'] = True
            mac = anycast.get('mac_address', None)
            if mac is None:
                new_anycast['mac_address'] = None
        new_want['anycast_address'] = new_anycast
        if not ac_address_only:
            hostname = want.get('hostname', None)
            if hostname is None:
                new_want['hostname'] = 'sonic'
            intf_name = want.get('interface_naming', None)
            if intf_name is None:
                new_want['interface_naming'] = 'native'
    return new_want