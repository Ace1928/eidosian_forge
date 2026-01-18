from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def execute_show_zone_status_cmd(self):
    command = 'show zone status vsan ' + str(self.vsan)
    output = execute_show_command(command, self.module)[0]
    return output