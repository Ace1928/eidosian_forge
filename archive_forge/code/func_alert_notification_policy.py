from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def alert_notification_policy(module):
    body = {'routes': module.params['routes'], 'Continue': module.params['Continue'], 'groupByStr': module.params['groupByStr'], 'muteTimeIntervals': module.params['muteTimeIntervals'], 'receiver': module.params['rootPolicyReceiver'], 'group_interval': module.params['groupInterval'], 'group_wait': module.params['groupWait'], 'object_matchers': module.params['objectMatchers'], 'repeat_interval': module.params['repeatInterval']}
    if module.params['grafana_url'][-1] == '/':
        module.params['grafana_url'] = module.params['grafana_url'][:-1]
    api_url = module.params['grafana_url'] + '/api/v1/provisioning/policies'
    result = requests.get(api_url, headers={'Authorization': 'Bearer ' + module.params['grafana_api_key']})
    if 'routes' not in result.json():
        api_url = module.params['grafana_url'] + '/api/v1/provisioning/policies'
        result = requests.put(api_url, json=body, headers={'Authorization': 'Bearer ' + module.params['grafana_api_key']})
        if result.status_code == 202:
            return (False, True, result.json())
        else:
            return (True, False, {'status': result.status_code, 'response': result.json()['message']})
    elif result.json()['receiver'] == module.params['rootPolicyReceiver'] and result.json()['routes'] == module.params['routes'] and (result.json()['group_wait'] == module.params['groupWait']) and (result.json()['group_interval'] == module.params['groupInterval']) and (result.json()['repeat_interval'] == module.params['repeatInterval']):
        return (False, False, result.json())
    else:
        api_url = module.params['grafana_url'] + '/api/v1/provisioning/policies'
        result = requests.put(api_url, json=body, headers={'Authorization': 'Bearer ' + module.params['grafana_api_key']})
        if result.status_code == 202:
            return (False, True, result.json())
        else:
            return (True, False, {'status': result.status_code, 'response': result.json()['message']})