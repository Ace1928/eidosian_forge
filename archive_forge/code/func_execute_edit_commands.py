from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
def execute_edit_commands(self, commands, arguments):
    arguments = arguments or []
    cmd = [self.nmcli_bin, 'con', 'edit'] + arguments
    data = '\n'.join(commands)
    return self.execute_command(cmd, data=data)