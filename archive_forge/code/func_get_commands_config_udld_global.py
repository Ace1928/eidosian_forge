from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_commands_config_udld_global(delta, reset, existing):
    commands = []
    for param, value in delta.items():
        if param == 'aggressive':
            command = 'udld aggressive' if value == 'enabled' else 'no udld aggressive'
            commands.append(command)
        elif param == 'msg_time':
            if value == 'default':
                if existing.get('msg_time') != PARAM_TO_DEFAULT_KEYMAP.get('msg_time'):
                    commands.append('no udld message-time')
            else:
                commands.append('udld message-time ' + value)
    if reset:
        command = 'udld reset'
        commands.append(command)
    return commands