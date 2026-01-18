from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def config_nets_export_host_addr(self):
    """Configures the destination IP address and destination UDP port number"""
    if is_ipv4_addr(self.host_ip):
        if self.type == 'ip':
            cmd = 'netstream export ip host %s %s' % (self.host_ip, self.host_port)
        else:
            cmd = 'netstream export vxlan inner-ip host %s %s' % (self.host_ip, self.host_port)
    elif self.type == 'ip':
        cmd = 'netstream export ip host ipv6 %s %s' % (self.host_ip, self.host_port)
    else:
        cmd = 'netstream export vxlan inner-ip host ipv6 %s %s' % (self.host_ip, self.host_port)
    if self.host_vpn:
        cmd += ' vpn-instance %s' % self.host_vpn
    if is_config_exist(self.config, cmd):
        self.exist_conf['host_ip'] = self.host_ip
        self.exist_conf['host_port'] = self.host_port
        if self.host_vpn:
            self.exist_conf['host_vpn'] = self.host_vpn
        if self.state == 'present':
            return
        else:
            undo = True
    elif self.state == 'absent':
        return
    else:
        undo = False
    self.cli_add_command(cmd, undo)