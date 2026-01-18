from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible.module_utils.connection import ConnectionError
from copy import deepcopy
def get_delete_all_bgp_neighbor_requests(self, commands):
    requests = []
    for cmd in commands:
        if cmd.get('neighbors', None):
            requests.extend(self.get_delete_vrf_specific_neighbor_request(cmd['vrf_name'], cmd['neighbors']))
        if 'peer_group' in cmd and cmd['peer_group']:
            for each in cmd['peer_group']:
                requests.append(self.get_delete_vrf_specific_peergroup_request(cmd['vrf_name'], each['name']))
    return requests