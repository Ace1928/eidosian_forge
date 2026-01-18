from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_bgp_neighbors_af_requests(self, commands, have, is_delete_all):
    requests = []
    for cmd in commands:
        vrf_name = cmd['vrf_name']
        as_val = cmd['bgp_as']
        match = None
        if not is_delete_all:
            match = next((have_cfg for have_cfg in have if have_cfg['vrf_name'] == vrf_name and have_cfg['bgp_as'] == as_val), None)
        requests.extend(self.get_delete_single_bgp_neighbors_af_request(cmd, is_delete_all, match))
    return requests