from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
from ansible.module_utils.connection import exec_command
def config_vbdif(self):
    """configure command at the VBDIF interface view"""
    if not self.vbdif_name:
        return
    vbdif_cmd = 'interface %s' % self.vbdif_name.lower().capitalize()
    exist = is_config_exist(self.config, vbdif_cmd)
    if not exist:
        self.module.fail_json(msg='Error: Interface %s is not exist.' % self.vbdif_name)
    vbdif_view = False
    if self.vbdif_bind_vpn:
        cmd = 'ip binding vpn-instance %s' % self.vbdif_bind_vpn
        exist = is_config_exist(self.config, cmd)
        if self.state == 'present' and (not exist):
            if not vbdif_view:
                self.cli_add_command(vbdif_cmd)
                vbdif_view = True
            self.cli_add_command(cmd)
        elif self.state == 'absent' and exist:
            if not vbdif_view:
                self.cli_add_command(vbdif_cmd)
                vbdif_view = True
            self.cli_add_command(cmd, undo=True)
    if self.arp_distribute_gateway:
        cmd = 'arp distribute-gateway enable'
        exist = is_config_exist(self.config, cmd)
        if self.arp_distribute_gateway == 'enable' and (not exist):
            if not vbdif_view:
                self.cli_add_command(vbdif_cmd)
                vbdif_view = True
            self.cli_add_command(cmd)
        elif self.arp_distribute_gateway == 'disable' and exist:
            if not vbdif_view:
                self.cli_add_command(vbdif_cmd)
                vbdif_view = True
            self.cli_add_command(cmd, undo=True)
    if self.arp_direct_route:
        cmd = 'arp direct-route enable'
        exist = is_config_exist(self.config, cmd)
        if self.arp_direct_route == 'enable' and (not exist):
            if not vbdif_view:
                self.cli_add_command(vbdif_cmd)
                vbdif_view = True
            self.cli_add_command(cmd)
        elif self.arp_direct_route == 'disable' and exist:
            if not vbdif_view:
                self.cli_add_command(vbdif_cmd)
                vbdif_view = True
            self.cli_add_command(cmd, undo=True)
    if self.vbdif_mac:
        cmd = 'mac-address %s' % self.vbdif_mac
        exist = is_config_exist(self.config, cmd)
        if self.state == 'present' and (not exist):
            if not vbdif_view:
                self.cli_add_command(vbdif_cmd)
                vbdif_view = True
            self.cli_add_command(cmd)
        elif self.state == 'absent' and exist:
            if not vbdif_view:
                self.cli_add_command(vbdif_cmd)
                vbdif_view = True
            self.cli_add_command('undo mac-address')
    if vbdif_view:
        self.cli_add_command('quit')