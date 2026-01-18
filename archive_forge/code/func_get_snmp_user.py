from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_snmp_user(user, module):
    command = 'show snmp user {0}'.format(user)
    body = execute_show_command(command, module, text=True)
    body_text = body[0]
    if 'No such entry' not in body[0]:
        body = execute_show_command(command, module)
    resource = {}
    try:
        if body[0].get('TABLE_snmp_user'):
            tablekey = 'TABLE_snmp_user'
            rowkey = 'ROW_snmp_user'
            tablegrpkey = 'TABLE_snmp_group_names'
            rowgrpkey = 'ROW_snmp_group_names'
            authkey = 'auth_protocol'
            privkey = 'priv_protocol'
            grpkey = 'group_names'
        elif body[0].get('TABLE_snmp_users'):
            tablekey = 'TABLE_snmp_users'
            rowkey = 'ROW_snmp_users'
            tablegrpkey = 'TABLE_groups'
            rowgrpkey = 'ROW_groups'
            authkey = 'auth'
            privkey = 'priv'
            grpkey = 'group'
        rt = body[0][tablekey][rowkey]
        if isinstance(rt, list):
            resource_table = rt[0]
        else:
            resource_table = rt
        resource['user'] = user
        resource['authentication'] = str(resource_table[authkey]).strip()
        encrypt = str(resource_table[privkey]).strip()
        if encrypt.startswith('aes'):
            resource['encrypt'] = 'aes-128'
        else:
            resource['encrypt'] = 'none'
        groups = []
        if tablegrpkey in resource_table:
            group_table = resource_table[tablegrpkey][rowgrpkey]
            try:
                for group in group_table:
                    groups.append(str(group[grpkey]).strip())
            except TypeError:
                groups.append(str(group_table[grpkey]).strip())
            if isinstance(rt, list):
                rt.pop(0)
                for each in rt:
                    groups.append(each['user'].strip())
        elif 'group' in resource_table:
            groups = resource_table['group']
            if isinstance(groups, str):
                groups = [groups]
        resource['group'] = groups
    except (KeyError, AttributeError, IndexError, TypeError):
        if not resource and body_text and ('No such entry' not in body_text):
            resource = get_non_structured_snmp_user(body_text)
    return resource