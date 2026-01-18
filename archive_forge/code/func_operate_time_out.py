from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def operate_time_out(self):
    """configure timeout parameters"""
    cmd = ''
    if self.timeout_type == 'manual':
        if self.type == 'ip':
            self.cli_add_command('quit')
            cmd = 'reset netstream cache ip slot %s' % self.manual_slot
            self.cli_add_command(cmd)
        elif self.type == 'vxlan':
            self.cli_add_command('quit')
            cmd = 'reset netstream cache vxlan inner-ip slot %s' % self.manual_slot
            self.cli_add_command(cmd)
    if not self.active_changed and (not self.inactive_changed) and (not self.tcp_changed):
        if self.commands:
            self.cli_load_config(self.commands)
            self.changed = True
        return
    if self.active_changed or self.inactive_changed:
        if self.type == 'ip':
            cmd = 'netstream timeout ip %s %s' % (self.timeout_type, self.timeout_interval)
        elif self.type == 'vxlan':
            cmd = 'netstream timeout vxlan inner-ip %s %s' % (self.timeout_type, self.timeout_interval)
        if self.state == 'absent':
            self.cli_add_command(cmd, undo=True)
        else:
            self.cli_add_command(cmd)
    if self.timeout_type == 'tcp-session' and self.tcp_changed:
        if self.type == 'ip':
            if self.state == 'present':
                cmd = 'netstream timeout ip tcp-session'
            else:
                cmd = 'undo netstream timeout ip tcp-session'
        elif self.type == 'vxlan':
            if self.state == 'present':
                cmd = 'netstream timeout vxlan inner-ip tcp-session'
            else:
                cmd = 'undo netstream timeout vxlan inner-ip tcp-session'
        self.cli_add_command(cmd)
    if self.commands:
        self.cli_load_config(self.commands)
        self.changed = True