from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_ntp_auth(module):
    command = 'show ntp authentication-status'
    body = execute_show_command(command, module)[0]
    ntp_auth_str = body['authentication']
    if 'enabled' in ntp_auth_str:
        ntp_auth = True
    else:
        ntp_auth = False
    return ntp_auth