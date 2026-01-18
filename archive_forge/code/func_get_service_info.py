from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
def get_service_info(module, auth, service):
    result = {'service_id': int(service['ID']), 'service_name': service['NAME'], 'group_id': int(service['GID']), 'group_name': service['GNAME'], 'owner_id': int(service['UID']), 'owner_name': service['UNAME'], 'state': STATES[service['TEMPLATE']['BODY']['state']]}
    roles_status = service['TEMPLATE']['BODY']['roles']
    roles = []
    for role in roles_status:
        nodes_ids = []
        if 'nodes' in role:
            for node in role['nodes']:
                nodes_ids.append(node['deploy_id'])
        roles.append({'name': role['name'], 'cardinality': role['cardinality'], 'state': STATES[int(role['state'])], 'ids': nodes_ids})
    result['roles'] = roles
    result['mode'] = int(parse_service_permissions(service))
    return result