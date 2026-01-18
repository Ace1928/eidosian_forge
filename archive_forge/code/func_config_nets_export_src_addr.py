from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def config_nets_export_src_addr(self):
    """Configures the source address for the exported packets"""
    if is_ipv4_addr(self.source_ip):
        if self.type == 'ip':
            cmd = 'netstream export ip source %s' % self.source_ip
        else:
            cmd = 'netstream export vxlan inner-ip source %s' % self.source_ip
    elif self.type == 'ip':
        cmd = 'netstream export ip source ipv6 %s' % self.source_ip
    else:
        cmd = 'netstream export vxlan inner-ip source ipv6 %s' % self.source_ip
    if is_config_exist(self.config, cmd):
        self.exist_conf['source_ip'] = self.source_ip
        if self.state == 'present':
            return
        else:
            undo = True
    elif self.state == 'absent':
        return
    else:
        undo = False
    self.cli_add_command(cmd, undo)