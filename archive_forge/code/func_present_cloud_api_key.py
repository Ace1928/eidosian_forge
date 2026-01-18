from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def present_cloud_api_key(module):
    body = {'name': module.params['name'], 'role': module.params['role']}
    api_url = 'https://grafana.com/api/orgs/' + module.params['org_slug'] + '/api-keys'
    result = requests.post(api_url, json=body, headers={'Authorization': 'Bearer ' + module.params['existing_cloud_api_key']})
    if result.status_code == 200:
        return (False, True, result.json())
    elif result.status_code == 409:
        return (module.params['fail_if_already_created'], False, 'A Cloud API key with the same name already exists')
    else:
        return (True, False, {'status': result.status_code, 'response': result.json()['message']})