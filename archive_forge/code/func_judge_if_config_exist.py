from __future__ import (absolute_import, division, print_function)
import re
import copy
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def judge_if_config_exist(self):
    """Judge whether configuration has existed"""
    if self.state == 'absent':
        if self.route_distinguisher or self.vpn_target_import or self.vpn_target_export or self.vpn_target_both:
            return False
        else:
            return True
    if self.evpn_info['evpn_inst'] != self.evpn:
        return False
    if self.evpn == 'disable' and self.evpn_info['evpn_inst'] == 'disable':
        return True
    if self.proposed['bridge_domain_id'] != self.existing['bridge_domain_id']:
        return False
    if self.proposed['route_distinguisher']:
        if self.proposed['route_distinguisher'] != self.existing['route_distinguisher']:
            return False
    if self.proposed['vpn_target_both']:
        if not self.existing['vpn_target_both']:
            return False
        if not set(self.proposed['vpn_target_both']).issubset(self.existing['vpn_target_both']):
            return False
    if self.proposed['vpn_target_import']:
        if not self.judge_if_vpn_target_exist('vpn_target_import'):
            return False
    if self.proposed['vpn_target_export']:
        if not self.judge_if_vpn_target_exist('vpn_target_export'):
            return False
    return True