from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_all_neighbors_af_modify_requests(self, match, conf_neighbors, vrf_name):
    requests = []
    for conf_neighbor in conf_neighbors:
        conf_neighbor_val = conf_neighbor.get('neighbor', None)
        if conf_neighbor_val:
            requests.extend(self.get_single_neighbors_af_modify_request(match, vrf_name, conf_neighbor_val, conf_neighbor))
    return requests