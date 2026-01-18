from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def build_merge_requests(self, conf):
    requests = []
    ip_neigh_config = dict()
    if 'ipv4_arp_timeout' in conf:
        ip_neigh_config['ipv4-arp-timeout'] = conf['ipv4_arp_timeout']
    if 'ipv4_drop_neighbor_aging_time' in conf:
        ip_neigh_config['ipv4-drop-neighbor-aging-time'] = conf['ipv4_drop_neighbor_aging_time']
    if 'ipv6_drop_neighbor_aging_time' in conf:
        ip_neigh_config['ipv6-drop-neighbor-aging-time'] = conf['ipv6_drop_neighbor_aging_time']
    if 'ipv6_nd_cache_expiry' in conf:
        ip_neigh_config['ipv6-nd-cache-expiry'] = conf['ipv6_nd_cache_expiry']
    if 'num_local_neigh' in conf:
        ip_neigh_config['num-local-neigh'] = conf['num_local_neigh']
    if ip_neigh_config:
        payload = {'config': ip_neigh_config}
        method = PATCH
        requests = {'path': CONFIG_URL, 'method': method, 'data': payload}
    return requests