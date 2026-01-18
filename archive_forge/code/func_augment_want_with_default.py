from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def augment_want_with_default(self, want):
    new_want = IP_NEIGH_CONFIG_DEFAULT
    if 'ipv4_arp_timeout' in want:
        new_want['ipv4_arp_timeout'] = want['ipv4_arp_timeout']
    if 'ipv4_drop_neighbor_aging_time' in want:
        new_want['ipv4_drop_neighbor_aging_time'] = want['ipv4_drop_neighbor_aging_time']
    if 'ipv6_drop_neighbor_aging_time' in want:
        new_want['ipv6_drop_neighbor_aging_time'] = want['ipv6_drop_neighbor_aging_time']
    if 'ipv6_nd_cache_expiry' in want:
        new_want['ipv6_nd_cache_expiry'] = want['ipv6_nd_cache_expiry']
    if 'num_local_neigh' in want:
        new_want['num_local_neigh'] = want['num_local_neigh']
    return new_want