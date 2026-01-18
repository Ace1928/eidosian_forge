from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def default_pim_interface_policies(existing, jp_bidir):
    commands = []
    if jp_bidir:
        if existing.get('jp_policy_in') or existing.get('jp_policy_out'):
            if existing.get('jp_type_in') == 'prefix':
                command = 'no ip pim jp-policy prefix-list {0}'.format(existing.get('jp_policy_in'))
        if command:
            commands.append(command)
    elif not jp_bidir:
        command = None
        for k in existing:
            if k == 'jp_policy_in':
                if existing.get('jp_policy_in'):
                    if existing.get('jp_type_in') == 'prefix':
                        command = 'no ip pim jp-policy prefix-list {0} in'.format(existing.get('jp_policy_in'))
                    else:
                        command = 'no ip pim jp-policy {0} in'.format(existing.get('jp_policy_in'))
            elif k == 'jp_policy_out':
                if existing.get('jp_policy_out'):
                    if existing.get('jp_type_out') == 'prefix':
                        command = 'no ip pim jp-policy prefix-list {0} out'.format(existing.get('jp_policy_out'))
                    else:
                        command = 'no ip pim jp-policy {0} out'.format(existing.get('jp_policy_out'))
            if command:
                commands.append(command)
            command = None
    if existing.get('neighbor_policy'):
        command = 'no ip pim neighbor-policy'
        commands.append(command)
    return commands