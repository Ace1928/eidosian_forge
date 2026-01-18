from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_create_vlan_map_request(self, configs, have):
    requests = []
    for conf in configs:
        new_vlan_map_list = conf.get('vlan_map', [])
        if new_vlan_map_list:
            for each_vlan_map in new_vlan_map_list:
                name = conf['name']
                vlan = each_vlan_map.get('vlan')
                vni = each_vlan_map.get('vni')
                matched = next((each_vxlan for each_vxlan in have if each_vxlan['name'] == name), None)
                is_change_needed = True
                if matched:
                    matched_vlan_map_list = matched.get('vlan_map', [])
                    if matched_vlan_map_list:
                        matched_vlan_map = next((e_vlan_map for e_vlan_map in matched_vlan_map_list if e_vlan_map['vni'] == vni), None)
                        if matched_vlan_map:
                            if matched_vlan_map['vlan'] == vlan:
                                is_change_needed = False
                if is_change_needed:
                    map_name = 'map_{0}_Vlan{1}'.format(vni, vlan)
                    payload = self.build_create_vlan_map_payload(conf, each_vlan_map)
                    url = 'data/sonic-vxlan:sonic-vxlan/VXLAN_TUNNEL_MAP'
                    request = {'path': url, 'method': PATCH, 'data': payload}
                    requests.append(request)
    return requests