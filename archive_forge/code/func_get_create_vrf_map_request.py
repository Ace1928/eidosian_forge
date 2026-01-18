from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_create_vrf_map_request(self, configs, have):
    requests = []
    for conf in configs:
        new_vrf_map_list = conf.get('vrf_map', [])
        if new_vrf_map_list:
            for each_vrf_map in new_vrf_map_list:
                name = conf['name']
                vrf = each_vrf_map.get('vrf')
                vni = each_vrf_map.get('vni')
                matched = next((each_vxlan for each_vxlan in have if each_vxlan['name'] == name), None)
                is_change_needed = True
                if matched:
                    matched_vrf_map_list = matched.get('vrf_map', [])
                    if matched_vrf_map_list:
                        matched_vrf_map = next((e_vrf_map for e_vrf_map in matched_vrf_map_list if e_vrf_map['vni'] == vni), None)
                        if matched_vrf_map:
                            if matched_vrf_map['vrf'] == vrf:
                                is_change_needed = False
                if is_change_needed:
                    payload = self.build_create_vrf_map_payload(conf, each_vrf_map)
                    url = 'data/sonic-vrf:sonic-vrf/VRF/VRF_LIST={vrf}/vni'.format(vrf=vrf)
                    request = {'path': url, 'method': PATCH, 'data': payload}
                    requests.append(request)
    return requests