from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def config_pim_interface(delta, existing, jp_bidir, isauth):
    command = None
    commands = []
    delta = fix_delta(delta, existing)
    if jp_bidir:
        if delta.get('jp_policy_in') or delta.get('jp_policy_out'):
            if existing.get('jp_type_in') == 'prefix':
                command = 'no ip pim jp-policy prefix-list {0}'.format(existing.get('jp_policy_in'))
            else:
                command = 'no ip pim jp-policy {0}'.format(existing.get('jp_policy_in'))
            if command:
                commands.append(command)
    for k, v in delta.items():
        if k in ['bfd', 'dr_prio', 'hello_interval', 'hello_auth_key', 'border', 'sparse']:
            if k == 'bfd':
                command = BFD_KEYMAP[v]
            elif v:
                command = PARAM_TO_COMMAND_KEYMAP.get(k).format(v)
            elif k == 'hello_auth_key':
                if isauth:
                    command = 'no ip pim hello-authentication ah-md5'
            else:
                command = 'no ' + PARAM_TO_COMMAND_KEYMAP.get(k).format(v)
            if command:
                commands.append(command)
        elif k in ['neighbor_policy', 'jp_policy_in', 'jp_policy_out', 'neighbor_type']:
            if k in ['neighbor_policy', 'neighbor_type']:
                temp = delta.get('neighbor_policy') or existing.get('neighbor_policy')
                if delta.get('neighbor_type') == 'prefix':
                    command = PARAM_TO_COMMAND_KEYMAP.get(k).format(temp)
                elif delta.get('neighbor_type') == 'routemap':
                    command = 'ip pim neighbor-policy {0}'.format(temp)
                elif existing.get('neighbor_type') == 'prefix':
                    command = PARAM_TO_COMMAND_KEYMAP.get(k).format(temp)
                elif existing.get('neighbor_type') == 'routemap':
                    command = 'ip pim neighbor-policy {0}'.format(temp)
            elif k in ['jp_policy_in', 'jp_type_in']:
                temp = delta.get('jp_policy_in') or existing.get('jp_policy_in')
                if delta.get('jp_type_in') == 'prefix':
                    command = PARAM_TO_COMMAND_KEYMAP.get(k).format(temp)
                elif delta.get('jp_type_in') == 'routemap':
                    command = 'ip pim jp-policy {0} in'.format(temp)
                elif existing.get('jp_type_in') == 'prefix':
                    command = PARAM_TO_COMMAND_KEYMAP.get(k).format(temp)
                elif existing.get('jp_type_in') == 'routemap':
                    command = 'ip pim jp-policy {0} in'.format(temp)
            elif k in ['jp_policy_out', 'jp_type_out']:
                temp = delta.get('jp_policy_out') or existing.get('jp_policy_out')
                if delta.get('jp_type_out') == 'prefix':
                    command = PARAM_TO_COMMAND_KEYMAP.get(k).format(temp)
                elif delta.get('jp_type_out') == 'routemap':
                    command = 'ip pim jp-policy {0} out'.format(temp)
                elif existing.get('jp_type_out') == 'prefix':
                    command = PARAM_TO_COMMAND_KEYMAP.get(k).format(temp)
                elif existing.get('jp_type_out') == 'routemap':
                    command = 'ip pim jp-policy {0} out'.format(temp)
            if command:
                commands.append(command)
        command = None
    if 'no ip pim sparse-mode' in commands:
        commands.remove('no ip pim sparse-mode')
        commands.append('no ip pim sparse-mode')
    return commands