from __future__ import absolute_import, division, print_function
import string
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def execute_show_cmd(self, cmd):
    output = execute_show_command(cmd, self.module)[0]
    return output