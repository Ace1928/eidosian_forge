from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_single_bgp_neighbors_af_request(self, conf, is_delete_all, match=None):
    requests = []
    vrf_name = conf['vrf_name']
    conf_neighbors = conf.get('neighbors', [])
    if match and (not conf_neighbors):
        conf_neighbors = match.get('neighbors', [])
        if conf_neighbors:
            conf_neighbors = [{'neighbor': nei['neighbor']} for nei in conf_neighbors]
    if not conf_neighbors:
        return requests
    mat_neighbors = None
    if match:
        mat_neighbors = match.get('neighbors', [])
    for conf_neighbor in conf_neighbors:
        conf_neighbor_val = conf_neighbor.get('neighbor', None)
        if not conf_neighbor_val:
            continue
        mat_neighbor = None
        if mat_neighbors:
            mat_neighbor = next((e_nei for e_nei in mat_neighbors if e_nei['neighbor'] == conf_neighbor_val), None)
        conf_nei_addr_fams = conf_neighbor.get('address_family', None)
        if mat_neighbor and (not conf_nei_addr_fams):
            conf_nei_addr_fams = mat_neighbor.get('address_family', None)
            if conf_nei_addr_fams:
                conf_nei_addr_fams = [{'afi': af['afi'], 'safi': af['safi']} for af in conf_nei_addr_fams]
        if not conf_nei_addr_fams:
            continue
        mat_nei_addr_fams = None
        if mat_neighbor:
            mat_nei_addr_fams = mat_neighbor.get('address_family', None)
        requests.extend(self.process_neighbor_delete_address_families(vrf_name, conf_nei_addr_fams, mat_nei_addr_fams, conf_neighbor_val, is_delete_all))
    return requests