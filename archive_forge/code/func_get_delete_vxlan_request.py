from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_vxlan_request(self, configs, have):
    requests = []
    if not configs:
        return requests
    vrf_map_requests = []
    vlan_map_requests = []
    src_ip_requests = []
    evpn_nvo_requests = []
    primary_ip_requests = []
    tunnel_requests = []
    for conf in configs:
        name = conf['name']
        src_ip = conf.get('source_ip', None)
        evpn_nvo = conf.get('evpn_nvo', None)
        primary_ip = conf.get('primary_ip', None)
        vlan_map_list = conf.get('vlan_map', None)
        vrf_map_list = conf.get('vrf_map', None)
        have_vlan_map_count = 0
        have_vrf_map_count = 0
        matched = next((each_vxlan for each_vxlan in have if each_vxlan['name'] == name), None)
        if matched:
            have_vlan_map = matched.get('vlan_map', [])
            have_vrf_map = matched.get('vrf_map', [])
            if have_vlan_map:
                have_vlan_map_count = len(have_vlan_map)
            if have_vrf_map:
                have_vrf_map_count = len(have_vrf_map)
        is_delete_full = False
        if name and vlan_map_list is None and (vrf_map_list is None) and (src_ip is None) and (evpn_nvo is None) and (primary_ip is None):
            is_delete_full = True
            vrf_map_list = matched.get('vrf_map', [])
            vlan_map_list = matched.get('vlan_map', [])
        if vlan_map_list is not None and len(vlan_map_list) == 0 and matched:
            vlan_map_list = matched.get('vlan_map', [])
        if vrf_map_list is not None and len(vrf_map_list) == 0 and matched:
            vrf_map_list = matched.get('vrf_map', [])
        if vrf_map_list:
            temp_vrf_map_requests = self.get_delete_vrf_map_request(conf, matched, name, vrf_map_list)
            if temp_vrf_map_requests:
                vrf_map_requests.extend(temp_vrf_map_requests)
                have_vrf_map_count -= len(temp_vrf_map_requests)
        if vlan_map_list:
            temp_vlan_map_requests = self.get_delete_vlan_map_request(conf, matched, name, vlan_map_list)
            if temp_vlan_map_requests:
                vlan_map_requests.extend(temp_vlan_map_requests)
                have_vlan_map_count -= len(temp_vlan_map_requests)
        if src_ip:
            src_ip_requests.extend(self.get_delete_src_ip_request(conf, matched, name, src_ip))
        if evpn_nvo:
            evpn_nvo_requests.extend(self.get_delete_evpn_request(conf, matched, evpn_nvo))
        if primary_ip:
            primary_ip_requests.extend(self.get_delete_primary_ip_request(conf, matched, name, primary_ip))
        if is_delete_full:
            tunnel_requests.extend(self.get_delete_tunnel_request(conf, matched, name))
    if vrf_map_requests:
        requests.extend(vrf_map_requests)
    if vlan_map_requests:
        requests.extend(vlan_map_requests)
    if src_ip_requests:
        requests.extend(src_ip_requests)
    if evpn_nvo_requests:
        requests.extend(evpn_nvo_requests)
    if primary_ip_requests:
        requests.extend(primary_ip_requests)
    if tunnel_requests:
        requests.extend(tunnel_requests)
    return requests