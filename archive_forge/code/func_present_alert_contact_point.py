from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def present_alert_contact_point(module):
    body = {'Name': module.params['name'], 'UID': module.params['uid'], 'type': module.params['type'], 'settings': module.params['settings'], 'DisableResolveMessage': module.params['disableResolveMessage']}
    if module.params['grafana_url'][-1] == '/':
        module.params['grafana_url'] = module.params['grafana_url'][:-1]
    api_url = module.params['grafana_url'] + '/api/v1/provisioning/contact-points'
    result = requests.post(api_url, json=body, headers={'Authorization': 'Bearer ' + module.params['grafana_api_key']})
    if result.status_code == 202:
        return (False, True, result.json())
    elif result.status_code == 500:
        sameConfig = False
        contactPointInfo = {}
        api_url = module.params['grafana_url'] + '/api/v1/provisioning/contact-points'
        result = requests.get(api_url, headers={'Authorization': 'Bearer ' + module.params['grafana_api_key']})
        for contact_points in result.json():
            if contact_points['uid'] == module.params['uid']:
                if contact_points['name'] == module.params['name'] and contact_points['type'] == module.params['type'] and contact_points['settings'] and (contact_points['settings'] == module.params['settings']) and (contact_points['disableResolveMessage'] == module.params['disableResolveMessage']):
                    sameConfig = True
                    contactPointInfo = contact_points
        if sameConfig:
            return (False, False, contactPointInfo)
        else:
            api_url = module.params['grafana_url'] + '/api/v1/provisioning/contact-points/' + module.params['uid']
            result = requests.put(api_url, json=body, headers={'Authorization': 'Bearer ' + module.params['grafana_api_key']})
            if result.status_code == 202:
                api_url = module.params['grafana_url'] + '/api/v1/provisioning/contact-points'
                result = requests.get(api_url, headers={'Authorization': 'Bearer ' + module.params['grafana_api_key']})
                for contact_points in result.json():
                    if contact_points['uid'] == module.params['uid']:
                        return (False, True, contact_points)
            else:
                return (True, False, {'status': result.status_code, 'response': result.json()['message']})
    else:
        return (True, False, {'status': result.status_code, 'response': result.json()['message']})