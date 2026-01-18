from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_all_vxlan_request(self, have):
    requests = []
    vrf_map_requests = []
    vlan_map_requests = []
    src_ip_requests = []
    primary_ip_requests = []
    evpn_nvo_requests = []
    tunnel_requests = []
    for conf in have:
        name = conf['name']
        vlan_map_list = conf.get('vlan_map', [])
        vrf_map_list = conf.get('vrf_map', [])
        src_ip = conf.get('source_ip', None)
        primary_ip = conf.get('primary_ip', None)
        evpn_nvo = conf.get('evpn_nvo', None)
        if vrf_map_list:
            vrf_map_requests.extend(self.get_delete_vrf_map_request(conf, conf, name, vrf_map_list))
        if vlan_map_list:
            vlan_map_requests.extend(self.get_delete_vlan_map_request(conf, conf, name, vlan_map_list))
        if src_ip:
            src_ip_requests.extend(self.get_delete_src_ip_request(conf, conf, name, src_ip))
        if primary_ip:
            primary_ip_requests.extend(self.get_delete_primary_ip_request(conf, conf, name, primary_ip))
        if evpn_nvo:
            evpn_nvo_requests.extend(self.get_delete_evpn_request(conf, conf, evpn_nvo))
        tunnel_requests.extend(self.get_delete_tunnel_request(conf, conf, name))
    if vrf_map_requests:
        requests.extend(vrf_map_requests)
    if vlan_map_requests:
        requests.extend(vlan_map_requests)
    if src_ip_requests:
        requests.extend(src_ip_requests)
    if primary_ip_requests:
        requests.extend(primary_ip_requests)
    if evpn_nvo_requests:
        requests.extend(evpn_nvo_requests)
    if tunnel_requests:
        requests.extend(tunnel_requests)
    return requests