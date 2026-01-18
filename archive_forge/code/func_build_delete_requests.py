from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def build_delete_requests(self, conf, delete_all):
    requests = []
    method = DELETE
    if delete_all:
        request = {'path': URL, 'method': method}
        requests.append(request)
        return requests
    if 'ipv4_arp_timeout' in conf:
        req_url = CONFIG_URL + '/ipv4-arp-timeout'
        request = {'path': req_url, 'method': method}
        requests.append(request)
    if 'ipv4_drop_neighbor_aging_time' in conf:
        req_url = CONFIG_URL + '/ipv4-drop-neighbor-aging-time'
        request = {'path': req_url, 'method': method}
        requests.append(request)
    if 'ipv6_drop_neighbor_aging_time' in conf:
        req_url = CONFIG_URL + '/ipv6-drop-neighbor-aging-time'
        request = {'path': req_url, 'method': method}
        requests.append(request)
    if 'ipv6_nd_cache_expiry' in conf:
        req_url = CONFIG_URL + '/ipv6-nd-cache-expiry'
        request = {'path': req_url, 'method': method}
        requests.append(request)
    if 'num_local_neigh' in conf:
        req_url = CONFIG_URL + '/num-local-neigh'
        request = {'path': req_url, 'method': method}
        requests.append(request)
    return requests