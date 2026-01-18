from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def present_cloud_stack(module):
    if not module.params['url']:
        module.params['url'] = 'https://' + module.params['stack_slug'] + '.grafana.net'
    body = {'name': module.params['name'], 'slug': module.params['stack_slug'], 'region': module.params['region'], 'url': module.params['url']}
    api_url = 'https://grafana.com/api/instances'
    result = requests.post(api_url, json=body, headers={'Authorization': 'Bearer ' + module.params['cloud_api_key']})
    if result.status_code == 200:
        return (False, True, result.json())
    elif result.status_code in [409, 403] and result.json()['message'] in ['That url is not available', 'Hosted instance limit reached']:
        stack_found = False
        if result.json['message'] == 'That url is not available':
            api_url = 'https://grafana.com/api/orgs/' + module.params['org_slug'] + '/instances'
            result = requests.get(api_url, headers={'Authorization': 'Bearer ' + module.params['cloud_api_key']})
            stackInfo = {}
            for stack in result.json()['items']:
                if stack['slug'] == module.params['stack_slug']:
                    stack_found = True
                    stackInfo = stack
            if stack_found:
                return (False, False, stackInfo)
            else:
                return (True, False, 'Stack is not found under your org')
        elif result.json['message'] == 'Hosted instance limit reached':
            return (True, False, 'You have reached Maximum number of Cloud Stacks in your Org.')
    else:
        return (True, False, {'status': result.status_code, 'response': result.json()['message']})