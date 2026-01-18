from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def present_cloud_plugin(module):
    body = {'plugin': module.params['name'], 'version': module.params['version']}
    api_url = 'https://grafana.com/api/instances/' + module.params['stack_slug'] + '/plugins'
    result = requests.post(api_url, json=body, headers={'Authorization': 'Bearer ' + module.params['cloud_api_key']})
    if result.status_code == 200:
        return (False, True, result.json())
    elif result.status_code == 409:
        api_url = 'https://grafana.com/api/instances/' + module.params['stack_slug'] + '/plugins/' + module.params['name']
        result = requests.get(api_url, headers={'Authorization': 'Bearer ' + module.params['cloud_api_key']})
        if result.json()['pluginSlug'] == module.params['name'] and result.json()['version'] == module.params['version']:
            return (False, False, result.json())
        else:
            api_url = 'https://grafana.com/api/instances/' + module.params['stack_slug'] + '/plugins/' + module.params['name']
            result = requests.post(api_url, json={'version': module.params['version']}, headers={'Authorization': 'Bearer ' + module.params['cloud_api_key']})
            return (False, True, result.json())
    else:
        return (True, False, {'status': result.status_code, 'response': result.json()['message']})