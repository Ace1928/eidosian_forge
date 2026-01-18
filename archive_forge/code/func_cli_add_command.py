from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, exec_command, run_commands
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import ComplexList
def cli_add_command(self, command, undo=False):
    """add command to self.update_cmd and self.commands"""
    self.commands.append('return')
    self.commands.append('mmi-mode enable')
    if self.action == 'commit':
        self.commands.append('sys')
    self.commands.append(command)
    self.updates_cmd.append(command)