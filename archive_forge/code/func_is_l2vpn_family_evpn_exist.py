from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config, ce_argument_spec
def is_l2vpn_family_evpn_exist(self):
    """Judge whether BGP-EVPN address family view has existed"""
    view_cmd = 'l2vpn-family evpn'
    return is_config_exist(self.config, view_cmd)