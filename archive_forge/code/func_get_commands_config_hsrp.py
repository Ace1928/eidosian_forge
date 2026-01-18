from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_commands_config_hsrp(delta, interface, args, existing):
    commands = []
    config_args = {'group': 'hsrp {group}', 'priority': '{priority}', 'preempt': '{preempt}', 'vip': '{vip}'}
    preempt = delta.get('preempt', None)
    group = delta.get('group', None)
    vip = delta.get('vip', None)
    priority = delta.get('priority', None)
    if preempt:
        if preempt == 'enabled':
            delta['preempt'] = 'preempt'
        elif preempt == 'disabled':
            delta['preempt'] = 'no preempt'
    if priority:
        if priority == 'default':
            if existing and existing.get('priority') != PARAM_TO_DEFAULT_KEYMAP.get('priority'):
                delta['priority'] = 'no priority'
            else:
                del delta['priority']
        else:
            delta['priority'] = 'priority {0}'.format(delta['priority'])
    if vip:
        if vip == 'default':
            if existing and existing.get('vip') != PARAM_TO_DEFAULT_KEYMAP.get('vip'):
                delta['vip'] = 'no ip'
            else:
                del delta['vip']
        else:
            delta['vip'] = 'ip {0}'.format(delta['vip'])
    for key in delta:
        command = config_args.get(key, 'DNE').format(**delta)
        if command and command != 'DNE':
            if key == 'group':
                commands.insert(0, command)
            else:
                commands.append(command)
        command = None
    auth_type = delta.get('auth_type', None)
    auth_string = delta.get('auth_string', None)
    auth_enc = delta.get('auth_enc', None)
    if auth_type or auth_string:
        if not auth_type:
            auth_type = args['auth_type']
        elif not auth_string:
            auth_string = args['auth_string']
        if auth_string != 'default':
            if auth_type == 'md5':
                command = 'authentication md5 key-string {0} {1}'.format(auth_enc, auth_string)
                commands.append(command)
            elif auth_type == 'text':
                command = 'authentication text {0}'.format(auth_string)
                commands.append(command)
        elif existing and existing.get('auth_string') != PARAM_TO_DEFAULT_KEYMAP.get('auth_string'):
            commands.append('no authentication')
    if commands and (not group):
        commands.insert(0, 'hsrp {0}'.format(args['group']))
    version = delta.get('version', None)
    if version:
        if version == '2':
            command = 'hsrp version 2'
        elif version == '1':
            command = 'hsrp version 1'
        commands.insert(0, command)
        commands.insert(0, 'interface {0}'.format(interface))
    if commands:
        if not commands[0].startswith('interface'):
            commands.insert(0, 'interface {0}'.format(interface))
    return commands