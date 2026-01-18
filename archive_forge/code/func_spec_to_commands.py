from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import (
def spec_to_commands(want, have):
    commands = []
    state = want.pop('state')
    if state == 'absent' and all((v is None for v in want.values())):
        for key in have:
            commands.append('delete system %s' % spec_key_to_device_key(key))
    for key in want:
        if want[key] is None:
            continue
        current = have.get(key)
        proposed = want[key]
        device_key = spec_key_to_device_key(key)
        if key in ['domain_search', 'name_server']:
            if not proposed:
                commands.append('delete system %s' % device_key)
            for config in proposed:
                if state == 'absent' and config in current:
                    commands.append("delete system %s '%s'" % (device_key, config))
                elif state == 'present' and config not in current:
                    commands.append("set system %s '%s'" % (device_key, config))
        elif state == 'absent' and current and proposed:
            commands.append('delete system %s' % device_key)
        elif state == 'present' and proposed and (proposed != current):
            commands.append("set system %s '%s'" % (device_key, proposed))
    return commands