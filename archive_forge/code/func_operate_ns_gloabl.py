from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_connection, rm_config_prefix
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def operate_ns_gloabl(self):
    """configure netstream global parameters"""
    cmd = ''
    if not self.sampler_changed and (not self.statistic_changed) and (not self.flexible_changed) and (not self.index_switch_changed):
        self.changed = False
        return
    if self.sampler_changed is True:
        if self.type == 'vxlan':
            self.module.fail_json(msg='Error: Netstream do not support vxlan sampler.')
        if self.interface != 'all':
            cmd = 'interface %s' % self.interface
            self.cli_add_command(cmd)
        cmd = 'netstream sampler random-packets %s %s' % (self.sampler_interval, self.sampler_direction)
        if self.state == 'present':
            self.cli_add_command(cmd)
        else:
            self.cli_add_command(cmd, undo=True)
        if self.interface != 'all':
            cmd = 'quit'
            self.cli_add_command(cmd)
    if self.statistic_changed is True:
        if self.interface != 'all':
            cmd = 'interface %s' % self.interface
            self.cli_add_command(cmd)
        cmd = 'netstream %s ip' % self.statistics_direction
        if self.state == 'present':
            self.cli_add_command(cmd)
        else:
            self.cli_add_command(cmd, undo=True)
        if self.interface != 'all':
            cmd = 'quit'
            self.cli_add_command(cmd)
    if self.flexible_changed is True:
        if self.interface != 'all':
            cmd = 'interface %s' % self.interface
            self.cli_add_command(cmd)
        if self.state == 'present':
            for statistic_tmp in self.existing['flexible_statistic']:
                tmp_list = statistic_tmp['statistics_record']
                if self.type == statistic_tmp['type']:
                    if self.type == 'ip':
                        if len(tmp_list) > 0:
                            cmd = 'netstream record %s ip' % tmp_list[0]
                            self.cli_add_command(cmd, undo=True)
                        cmd = 'netstream record %s ip' % self.statistics_record
                        self.cli_add_command(cmd)
                    if self.type == 'vxlan':
                        if len(tmp_list) > 0:
                            cmd = 'netstream record %s vxlan inner-ip' % tmp_list[0]
                            self.cli_add_command(cmd, undo=True)
                        cmd = 'netstream record %s vxlan inner-ip' % self.statistics_record
                        self.cli_add_command(cmd)
        else:
            if self.type == 'ip':
                cmd = 'netstream record %s ip' % self.statistics_record
                self.cli_add_command(cmd, undo=True)
            if self.type == 'vxlan':
                cmd = 'netstream record %s vxlan inner-ip' % self.statistics_record
                self.cli_add_command(cmd, undo=True)
        if self.interface != 'all':
            cmd = 'quit'
            self.cli_add_command(cmd)
    if self.index_switch_changed is True:
        if self.interface != 'all':
            self.module.fail_json(msg='Error: Index-switch function should be used globally.')
        if self.type == 'ip':
            cmd = 'netstream export ip index-switch %s' % self.index_switch
        else:
            cmd = 'netstream export vxlan inner-ip index-switch %s' % self.index_switch
        if self.state == 'present':
            self.cli_add_command(cmd)
        else:
            self.cli_add_command(cmd, undo=True)
    if self.commands:
        self.cli_load_config(self.commands)
        self.changed = True