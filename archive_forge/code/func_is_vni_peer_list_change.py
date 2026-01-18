from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_vni_peer_list_change(self, nve_name, vni_id, peer_ip_list):
    """is vni peer list change"""
    if not self.nve_info:
        return True
    if self.nve_info['ifName'] == nve_name:
        if not self.nve_info['vni_peer_ips']:
            return True
        nve_peer_info = list()
        for nve_peer in self.nve_info['vni_peer_ips']:
            if nve_peer['vniId'] == vni_id:
                nve_peer_info.append(nve_peer)
        if not nve_peer_info:
            return True
        nve_peer_list = nve_peer_info[0]['peerAddr']
        for peer in peer_ip_list:
            if peer not in nve_peer_list:
                return True
        return False