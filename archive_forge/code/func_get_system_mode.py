from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_system_mode(module):
    command = {'command': 'show system mode', 'output': 'text'}
    body = run_commands(module, [command])[0]
    if body and 'normal' in body.lower():
        mode = 'normal'
    else:
        mode = 'maintenance'
    return mode