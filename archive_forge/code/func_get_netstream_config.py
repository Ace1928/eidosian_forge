from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def get_netstream_config(self):
    """get current netstream configuration"""
    cmd = 'display current-configuration | include ^netstream export'
    rc, out, err = exec_command(self.module, cmd)
    if rc != 0:
        self.module.fail_json(msg=err)
    config = str(out).strip()
    return config