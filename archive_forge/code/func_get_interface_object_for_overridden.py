from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_interface_object_for_overridden(self, have):
    objects = list()
    for obj in have:
        if 'name' in obj and obj['name'] != 'Management0':
            ipv4_addresses = obj['ipv4']['addresses']
            ipv6_addresses = obj['ipv6']['addresses']
            anycast_addresses = obj['ipv4']['anycast_addresses']
            ipv6_enable = obj['ipv6']['enabled']
            if ipv4_addresses is not None or ipv6_addresses is not None:
                objects.append(obj.copy())
                continue
            if ipv6_enable or anycast_addresses is not None:
                objects.append(obj.copy())
                continue
    return objects