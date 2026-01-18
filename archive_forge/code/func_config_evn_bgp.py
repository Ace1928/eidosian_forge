from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
from ansible.module_utils.connection import exec_command
def config_evn_bgp(self):
    """enables EVN BGP and configure evn bgp command"""
    evn_bgp_view = False
    evn_bgp_enable = False
    cmd = 'evn bgp'
    exist = is_config_exist(self.config, cmd)
    if self.evn_bgp == 'enable' or exist:
        evn_bgp_enable = True
    if self.evn_bgp:
        if self.evn_bgp == 'enable' and (not exist):
            self.cli_add_command(cmd)
            evn_bgp_view = True
        elif self.evn_bgp == 'disable' and exist:
            self.cli_add_command(cmd, undo=True)
            return
    if evn_bgp_enable and self.evn_source_ip:
        cmd = 'source-address %s' % self.evn_source_ip
        exist = is_config_exist(self.config, cmd)
        if self.state == 'present' and (not exist):
            if not evn_bgp_view:
                self.cli_add_command('evn bgp')
                evn_bgp_view = True
            self.cli_add_command(cmd)
        elif self.state == 'absent' and exist:
            if not evn_bgp_view:
                self.cli_add_command('evn bgp')
                evn_bgp_view = True
            self.cli_add_command(cmd, undo=True)
    if evn_bgp_enable and self.evn_peer_ip:
        cmd = 'peer %s' % self.evn_peer_ip
        exist = is_config_exist(self.config, cmd)
        if self.state == 'present':
            if not exist:
                if not evn_bgp_view:
                    self.cli_add_command('evn bgp')
                    evn_bgp_view = True
                self.cli_add_command(cmd)
                if self.evn_reflect_client == 'enable':
                    self.cli_add_command('peer %s reflect-client' % self.evn_peer_ip)
            elif self.evn_reflect_client:
                cmd = 'peer %s reflect-client' % self.evn_peer_ip
                exist = is_config_exist(self.config, cmd)
                if self.evn_reflect_client == 'enable' and (not exist):
                    if not evn_bgp_view:
                        self.cli_add_command('evn bgp')
                        evn_bgp_view = True
                    self.cli_add_command(cmd)
                elif self.evn_reflect_client == 'disable' and exist:
                    if not evn_bgp_view:
                        self.cli_add_command('evn bgp')
                        evn_bgp_view = True
                    self.cli_add_command(cmd, undo=True)
        elif exist:
            if not evn_bgp_view:
                self.cli_add_command('evn bgp')
                evn_bgp_view = True
            self.cli_add_command(cmd, undo=True)
    if evn_bgp_enable and self.evn_server:
        cmd = 'server enable'
        exist = is_config_exist(self.config, cmd)
        if self.evn_server == 'enable' and (not exist):
            if not evn_bgp_view:
                self.cli_add_command('evn bgp')
                evn_bgp_view = True
            self.cli_add_command(cmd)
        elif self.evn_server == 'disable' and exist:
            if not evn_bgp_view:
                self.cli_add_command('evn bgp')
                evn_bgp_view = True
            self.cli_add_command(cmd, undo=True)
    if evn_bgp_view:
        self.cli_add_command('quit')