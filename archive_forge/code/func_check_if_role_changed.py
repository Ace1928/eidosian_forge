from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def check_if_role_changed(client, role, db_name, privileges, authenticationRestrictions, roles):
    role_dict = role_find(client, role, db_name)
    changed = False
    if role_dict:
        reformat_authenticationRestrictions = []
        if 'authenticationRestrictions' in role_dict:
            for item in role_dict['authenticationRestrictions']:
                reformat_authenticationRestrictions.append(item[0])
        if 'privileges' in role_dict and [{'resource': d['resource'], 'actions': sorted(d['actions'])} for d in role_dict['privileges']] != [{'resource': d['resource'], 'actions': sorted(d['actions'])} for d in privileges] or ('privileges' not in role_dict and privileges != []):
            changed = True
        elif 'roles' in role_dict and sorted(role_dict['roles'], key=lambda x: (x['db'], x['role'])) != sorted(roles, key=lambda x: (x['db'], x['role'])) or ('roles' not in role_dict and roles != []):
            changed = True
        elif 'authenticationRestrictions' in role_dict and sorted(reformat_authenticationRestrictions, key=lambda x: (x.get('clientSource', ''), x.get('serverAddress', ''))) != sorted(authenticationRestrictions, key=lambda x: (x.get('clientSource', ''), x.get('serverAddress', ''))) or ('authenticationRestrictions' not in role_dict and authenticationRestrictions != []):
            changed = True
    else:
        raise Exception('Role not found')
    return changed