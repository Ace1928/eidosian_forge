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
def get_delete_bgp_neighbor_requests(self, commands, have, want, is_delete_all):
    requests = []
    if is_delete_all:
        requests = self.get_delete_all_bgp_neighbor_requests(commands)
    else:
        for cmd in commands:
            vrf_name = cmd['vrf_name']
            as_val = cmd['bgp_as']
            neighbors = cmd.get('neighbors', None)
            peer_group = cmd.get('peer_group', None)
            want_match = next((cfg for cfg in want if vrf_name == cfg['vrf_name'] and as_val == cfg['bgp_as']), None)
            want_neighbors = want_match.get('neighbors', None)
            want_peer_group = want_match.get('peer_group', None)
            if neighbors is None and peer_group is None and (want_neighbors is None) and (want_peer_group is None):
                new_cmd = {}
                for each in have:
                    if vrf_name == each['vrf_name'] and as_val == each['bgp_as']:
                        new_neighbors = []
                        new_pg = []
                        if each.get('neighbors', None):
                            new_neighbors = [{'neighbor': i['neighbor']} for i in each.get('neighbors', None)]
                        if each.get('peer_group', None):
                            new_pg = [{'name': i['name']} for i in each.get('peer_group', None)]
                        if new_neighbors:
                            new_cmd['neighbors'] = new_neighbors
                            requests.extend(self.get_delete_vrf_specific_neighbor_request(vrf_name, new_cmd['neighbors']))
                        if new_pg:
                            new_cmd['name'] = new_pg
                            for each in new_cmd['name']:
                                requests.append(self.get_delete_vrf_specific_peergroup_request(vrf_name, each['name']))
                        break
            else:
                if neighbors:
                    requests.extend(self.get_delete_specific_bgp_param_request(vrf_name, cmd, want_match))
                if peer_group:
                    requests.extend(self.get_delete_specific_bgp_peergroup_param_request(vrf_name, cmd, want_match))
    return requests