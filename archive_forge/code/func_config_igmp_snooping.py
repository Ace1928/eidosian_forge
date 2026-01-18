from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def config_igmp_snooping(delta, existing, default=False):
    CMDS = {'snooping': 'ip igmp snooping', 'group_timeout': 'ip igmp snooping group-timeout {}', 'link_local_grp_supp': 'ip igmp snooping link-local-groups-suppression', 'v3_report_supp': 'ip igmp snooping v3-report-suppression', 'report_supp': 'ip igmp snooping report-suppression'}
    commands = []
    command = None
    gt_command = None
    for key, value in delta.items():
        if value:
            if default and key == 'group_timeout':
                if existing.get(key):
                    gt_command = 'no ' + CMDS.get(key).format(existing.get(key))
            elif value == 'default' and key == 'group_timeout':
                if existing.get(key):
                    command = 'no ' + CMDS.get(key).format(existing.get(key))
            else:
                command = CMDS.get(key).format(value)
        else:
            command = 'no ' + CMDS.get(key).format(value)
        if command:
            commands.append(command)
        command = None
    if gt_command:
        commands.append(gt_command)
    return commands