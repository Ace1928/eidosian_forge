from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def get_evpn_overlay_config(self):
    """get evpn-overlay enable configuration"""
    cmd = 'display current-configuration | include ^evpn-overlay enable'
    rc, out, err = exec_command(self.module, cmd)
    if rc != 0:
        self.module.fail_json(msg=err)
    return out