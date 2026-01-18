from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_vni_peer_list_exist(self, nve_name, vni_id, peer_ip):
    """is vni peer list exist"""
    if not self.nve_info:
        return False
    if self.nve_info['ifName'] == nve_name:
        for member in self.nve_info['vni_peer_ips']:
            if member['vniId'] == vni_id and peer_ip in member['peerAddr']:
                return True
    return False