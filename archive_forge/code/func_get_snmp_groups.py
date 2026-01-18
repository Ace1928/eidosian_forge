from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_snmp_groups(module):
    data = execute_show_command('show snmp group', module)[0]
    group_list = []
    try:
        group_table = data['TABLE_role']['ROW_role']
        for group in group_table:
            group_list.append(group['role_name'])
    except (KeyError, AttributeError):
        pass
    return group_list