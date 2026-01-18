from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def config_nets_export_vxlan_ver(self):
    """Configures the version for the exported packets carrying VXLAN flexible flow statistics"""
    cmd = 'netstream export vxlan inner-ip version 9'
    if is_config_exist(self.config, cmd):
        self.exist_conf['version'] = self.version
        if self.state == 'present':
            return
        else:
            undo = True
    elif self.state == 'absent':
        return
    else:
        undo = False
    self.cli_add_command(cmd, undo)