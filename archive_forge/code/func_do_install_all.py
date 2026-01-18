from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def do_install_all(module, issu, image, kick=None):
    """Perform the switch upgrade using the 'install all' command"""
    impact_data = check_mode(module, issu, image, kick)
    if module.check_mode:
        msg = '*** SWITCH WAS NOT UPGRADED: IMPACT DATA ONLY ***'
        impact_data['processed'].append(msg)
        return impact_data
    if impact_data['error']:
        return impact_data
    elif not impact_data['upgrade_needed']:
        return impact_data
    else:
        if impact_data['disruptive']:
            if issu == 'yes':
                msg = 'ISSU/ISSD requested but impact data indicates ISSU/ISSD is not possible'
                module.fail_json(msg=msg, raw_data=impact_data['list_data'])
            else:
                issu = 'no'
        commands = build_install_cmd_set(issu, image, kick, 'install')
        opts = {'ignore_timeout': True}
        upgrade = check_install_in_progress(module, commands, opts)
        if upgrade['invalid_command'] and 'force' in commands[1]:
            commands = build_install_cmd_set(issu, image, kick, 'install', False)
            upgrade = check_install_in_progress(module, commands, opts)
        upgrade['upgrade_cmd'] = commands
        if upgrade['server_error']:
            upgrade['upgrade_succeeded'] = True
            upgrade['use_impact_data'] = True
        if upgrade['use_impact_data']:
            if upgrade['upgrade_succeeded']:
                upgrade = impact_data
                upgrade['upgrade_succeeded'] = True
            else:
                upgrade = impact_data
                upgrade['upgrade_succeeded'] = False
        if not upgrade['upgrade_succeeded']:
            upgrade['error'] = True
    return upgrade