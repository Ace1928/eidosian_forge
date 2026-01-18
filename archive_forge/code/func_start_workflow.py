from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
import json
def start_workflow(module):
    """
    :param module:
    :return: response and msg
    """
    if module.params['https']:
        transport_protocol = 'https'
    else:
        transport_protocol = 'http'
    application_token = str(module.params['token_key'])
    url = str(transport_protocol) + '://' + str(module.params['iap_fqdn']) + ':' + str(module.params['iap_port']) + '/workflow_engine/startJobWithOptions/' + str(module.params['workflow_name']) + '?token=' + str(application_token)
    options = {'variables': module.params['variables'], 'description': str(module.params['description'])}
    payload = {'workflow': module.params['workflow_name'], 'options': options}
    json_body = module.jsonify(payload)
    headers = dict()
    headers['Content-Type'] = 'application/json'
    response, info = fetch_url(module, url, data=json_body, headers=headers)
    response_code = str(info['status'])
    if info['status'] not in [200, 201]:
        module.fail_json(msg='Failed to connect to Itential Automation Platform. Response code is ' + response_code)
    jsonResponse = json.loads(response.read().decode('utf-8'))
    module.exit_json(changed=True, msg={'workflow_name': module.params['workflow_name'], 'status': 'started'}, response=jsonResponse)