from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.vxlans.vxlans import VxlansArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def fill_vrf_map(self, vxlans, vxlan_vrf_list):
    for each_vrf in vxlan_vrf_list:
        vni = each_vrf.get('vni', None)
        if vni is None:
            continue
        matched_vtep = None
        for each_vxlan in vxlans:
            for each_vlan in each_vxlan.get('vlan_map', []):
                if vni == each_vlan['vni']:
                    matched_vtep = each_vxlan
        if matched_vtep:
            vni = int(each_vrf['vni'])
            vrf = each_vrf['vrf_name']
            vrf_map = matched_vtep.get('vrf_map')
            if vrf_map:
                vrf_map.append(dict({'vni': vni, 'vrf': vrf}))
            else:
                matched_vtep['vrf_map'] = [dict({'vni': vni, 'vrf': vrf})]