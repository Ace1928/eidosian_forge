from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def present_folder(module):
    if module.params['grafana_url'][-1] == '/':
        module.params['grafana_url'] = module.params['grafana_url'][:-1]
    body = {'uid': module.params['uid'], 'title': module.params['title']}
    api_url = module.params['grafana_url'] + '/api/folders'
    result = requests.post(api_url, json=body, headers={'Authorization': 'Bearer ' + module.params['grafana_api_key']})
    if result.status_code == 200:
        return (False, True, result.json())
    elif result.status_code == 412:
        sameConfig = False
        folderInfo = {}
        api_url = module.params['grafana_url'] + '/api/folders'
        result = requests.get(api_url, headers={'Authorization': 'Bearer ' + module.params['grafana_api_key']})
        for folder in result.json():
            if folder['uid'] == module.params['uid'] and folder['title'] == module.params['title']:
                sameConfig = True
                folderInfo = folder
        if sameConfig:
            return (False, False, folderInfo)
        else:
            body = {'uid': module.params['uid'], 'title': module.params['title'], 'overwrite': module.params['overwrite']}
            api_url = module.params['grafana_url'] + '/api/folders/' + module.params['uid']
            result = requests.put(api_url, json=body, headers={'Authorization': 'Bearer ' + module.params['grafana_api_key']})
            if result.status_code == 200:
                return (False, True, result.json())
            else:
                return (True, False, {'status': result.status_code, 'response': result.json()['message']})
    else:
        return (True, False, {'status': result.status_code, 'response': result.json()['message']})