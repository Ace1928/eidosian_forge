from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def igmp_snooping_gt_dependency(command, existing, module):
    gt = [i for i in command if i.startswith('ip igmp snooping group-timeout')]
    if gt:
        if 'no ip igmp snooping' in command or (existing['snooping'] is False and 'ip igmp snooping' not in command):
            msg = 'group-timeout cannot be enabled or changed when ip igmp snooping is disabled'
            module.fail_json(msg=msg)
        else:
            command.remove(gt[0])
            command.append(gt[0])