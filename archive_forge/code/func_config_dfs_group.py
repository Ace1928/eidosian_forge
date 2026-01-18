from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
from ansible.module_utils.connection import exec_command
def config_dfs_group(self):
    """manage Dynamic Fabric Service (DFS) group configuration"""
    if not self.dfs_id:
        return
    dfs_view = False
    view_cmd = 'dfs-group %s' % self.dfs_id
    exist = is_config_exist(self.config, view_cmd)
    if self.state == 'present' and (not exist):
        self.cli_add_command(view_cmd)
        dfs_view = True
    if self.state == 'absent' and exist:
        if not self.dfs_source_ip and (not self.dfs_udp_port) and (not self.dfs_all_active) and (not self.dfs_peer_ip):
            self.cli_add_command(view_cmd, undo=True)
            return
    if self.dfs_source_ip:
        cmd = 'source ip %s' % self.dfs_source_ip
        if self.dfs_source_vpn:
            cmd += ' vpn-instance %s' % self.dfs_source_vpn
        exist = is_config_exist(self.config, cmd)
        if self.state == 'present' and (not exist):
            if not dfs_view:
                self.cli_add_command(view_cmd)
                dfs_view = True
            self.cli_add_command(cmd)
        if self.state == 'absent' and exist:
            if not dfs_view:
                self.cli_add_command(view_cmd)
                dfs_view = True
            self.cli_add_command(cmd, undo=True)
    if self.dfs_udp_port:
        cmd = 'udp port %s' % self.dfs_udp_port
        exist = is_config_exist(self.config, cmd)
        if self.state == 'present' and (not exist):
            if not dfs_view:
                self.cli_add_command(view_cmd)
                dfs_view = True
            self.cli_add_command(cmd)
        elif self.state == 'absent' and exist:
            if not dfs_view:
                self.cli_add_command(view_cmd)
                dfs_view = True
            self.cli_add_command(cmd, undo=True)
    aa_cmd = 'active-active-gateway'
    aa_exist = is_config_exist(self.config, aa_cmd)
    aa_view = False
    if self.dfs_all_active == 'disable':
        if aa_exist:
            cmd = 'peer %s' % self.dfs_peer_ip
            if self.dfs_source_vpn:
                cmd += ' vpn-instance %s' % self.dfs_peer_vpn
            exist = is_config_exist(self.config, cmd)
            if self.state == 'absent' and exist:
                if not dfs_view:
                    self.cli_add_command(view_cmd)
                    dfs_view = True
                self.cli_add_command(aa_cmd)
                self.cli_add_command(cmd, undo=True)
                self.cli_add_command('quit')
            if not dfs_view:
                self.cli_add_command(view_cmd)
                dfs_view = True
            self.cli_add_command(aa_cmd, undo=True)
    elif self.dfs_all_active == 'enable':
        if not aa_exist:
            if not dfs_view:
                self.cli_add_command(view_cmd)
                dfs_view = True
            self.cli_add_command(aa_cmd)
            aa_view = True
        if self.dfs_peer_ip:
            cmd = 'peer %s' % self.dfs_peer_ip
            if self.dfs_peer_vpn:
                cmd += ' vpn-instance %s' % self.dfs_peer_vpn
            exist = is_config_exist(self.config, cmd)
            if self.state == 'present' and (not exist):
                if not dfs_view:
                    self.cli_add_command(view_cmd)
                    dfs_view = True
                if not aa_view:
                    self.cli_add_command(aa_cmd)
                self.cli_add_command(cmd)
                self.cli_add_command('quit')
            elif self.state == 'absent' and exist:
                if not dfs_view:
                    self.cli_add_command(view_cmd)
                    dfs_view = True
                if not aa_view:
                    self.cli_add_command(aa_cmd)
                self.cli_add_command(cmd, undo=True)
                self.cli_add_command('quit')
    elif aa_exist and self.dfs_peer_ip:
        cmd = 'peer %s' % self.dfs_peer_ip
        if self.dfs_peer_vpn:
            cmd += ' vpn-instance %s' % self.dfs_peer_vpn
        exist = is_config_exist(self.config, cmd)
        if self.state == 'present' and (not exist):
            if not dfs_view:
                self.cli_add_command(view_cmd)
                dfs_view = True
            self.cli_add_command(aa_cmd)
            self.cli_add_command(cmd)
            self.cli_add_command('quit')
        elif self.state == 'absent' and exist:
            if not dfs_view:
                self.cli_add_command(view_cmd)
                dfs_view = True
            self.cli_add_command(aa_cmd)
            self.cli_add_command(cmd, undo=True)
            self.cli_add_command('quit')
        else:
            pass
    elif not aa_exist and self.dfs_peer_ip and (self.state == 'present'):
        self.module.fail_json(msg='Error: All-active gateways is not enable.')
    else:
        pass
    if dfs_view:
        self.cli_add_command('quit')