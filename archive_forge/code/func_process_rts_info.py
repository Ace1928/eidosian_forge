from __future__ import (absolute_import, division, print_function)
import re
import copy
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def process_rts_info(self):
    """Process RTS information"""
    if not self.evpn_info['vpn_target_export'] or not self.evpn_info['vpn_target_import']:
        return
    vpn_target_export = copy.deepcopy(self.evpn_info['vpn_target_export'])
    for ele in vpn_target_export:
        if ele in self.evpn_info['vpn_target_import']:
            self.evpn_info['vpn_target_both'].append(ele)
            self.evpn_info['vpn_target_export'].remove(ele)
            self.evpn_info['vpn_target_import'].remove(ele)