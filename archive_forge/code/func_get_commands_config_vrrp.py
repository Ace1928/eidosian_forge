from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_commands_config_vrrp(delta, existing, group):
    commands = []
    CMDS = {'priority': 'priority {0}', 'preempt': 'preempt', 'vip': 'address {0}', 'interval': 'advertisement-interval {0}', 'auth': 'authentication text {0}', 'admin_state': '{0}'}
    for arg in ['vip', 'priority', 'interval', 'admin_state']:
        val = delta.get(arg)
        if val == 'default':
            val = PARAM_TO_DEFAULT_KEYMAP.get(arg)
            if val != existing.get(arg):
                commands.append(CMDS.get(arg).format(val))
        elif val:
            commands.append(CMDS.get(arg).format(val))
    preempt = delta.get('preempt')
    auth = delta.get('authentication')
    if preempt:
        commands.append(CMDS.get('preempt'))
    elif preempt is False:
        commands.append('no ' + CMDS.get('preempt'))
    if auth:
        if auth != 'default':
            commands.append(CMDS.get('auth').format(auth))
        elif existing.get('authentication'):
            commands.append('no authentication')
    if commands:
        commands.insert(0, 'vrrp {0}'.format(group))
    return commands