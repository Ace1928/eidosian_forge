from __future__ import (absolute_import, division, print_function)
import re
import copy
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def process_input_params(self):
    """Process input parameters"""
    if self.state == 'absent':
        self.evpn = None
    elif self.evpn == 'disable':
        return
    if self.vpn_target_both:
        for ele in self.vpn_target_both:
            if ele in self.vpn_target_export:
                self.vpn_target_export.remove(ele)
            if ele in self.vpn_target_import:
                self.vpn_target_import.remove(ele)
    if self.vpn_target_export and self.vpn_target_import:
        vpn_target_export = copy.deepcopy(self.vpn_target_export)
        for ele in vpn_target_export:
            if ele in self.vpn_target_import:
                self.vpn_target_both.append(ele)
                self.vpn_target_import.remove(ele)
                self.vpn_target_export.remove(ele)