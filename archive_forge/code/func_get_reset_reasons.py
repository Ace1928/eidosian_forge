from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_reset_reasons(module):
    command = {'command': 'show maintenance on-reload reset-reasons', 'output': 'text'}
    body = run_commands(module, [command])[0]
    return body