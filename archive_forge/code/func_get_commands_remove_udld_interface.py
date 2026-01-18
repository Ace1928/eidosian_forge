from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_commands_remove_udld_interface(delta, interface, module, existing):
    commands = []
    existing, mode_str = get_udld_interface(module, interface)
    mode = delta['mode']
    if mode == 'aggressive':
        command = 'no udld aggressive'
    elif mode == 'enabled':
        if mode_str == 'udld enable':
            command = 'no udld enable'
        else:
            command = 'udld disable'
    elif mode == 'disabled':
        if mode_str == 'no udld disable':
            command = 'udld disable'
        else:
            command = 'no udld enable'
    if command:
        commands.append(command)
        commands.insert(0, 'interface {0}'.format(interface))
    return commands