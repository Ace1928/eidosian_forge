from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def build_create_vlan_map_payload(self, conf, vlan_map):
    payload_url = dict()
    vlan_map_dict = dict()
    vlan_map_dict['name'] = conf['name']
    vlan_map_dict['mapname'] = 'map_{vni}_Vlan{vlan}'.format(vni=vlan_map['vni'], vlan=vlan_map['vlan'])
    vlan_map_dict['vlan'] = 'Vlan{vlan}'.format(vlan=vlan_map['vlan'])
    vlan_map_dict['vni'] = vlan_map['vni']
    payload_url['sonic-vxlan:VXLAN_TUNNEL_MAP'] = {'VXLAN_TUNNEL_MAP_LIST': [vlan_map_dict]}
    return payload_url