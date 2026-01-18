from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config, ce_argument_spec
def is_policy_vpn_target_exist(self):
    """Judge whether the VPN-Target filtering is enabled"""
    view_cmd = 'undo policy vpn-target'
    if is_config_exist(self.bgp_evpn_config, view_cmd):
        return False
    else:
        return True