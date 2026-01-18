from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.vxlans.vxlans import VxlansArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def fill_tunnel_source_ip(self, vxlans, vxlan_tunnels, vxlans_evpn_nvo_list):
    for each_tunnel in vxlan_tunnels:
        vxlan = dict()
        vxlan['name'] = each_tunnel['name']
        vxlan['source_ip'] = each_tunnel.get('src_ip', None)
        vxlan['primary_ip'] = each_tunnel.get('primary_ip', None)
        vxlan['evpn_nvo'] = None
        evpn_nvo = next((nvo_map['name'] for nvo_map in vxlans_evpn_nvo_list if nvo_map['source_vtep'] == vxlan['name']), None)
        if evpn_nvo:
            vxlan['evpn_nvo'] = evpn_nvo
        vxlans.append(vxlan)