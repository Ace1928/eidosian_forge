from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
def modify_baseline(module, rest_obj):
    name = module.params['names'][0]
    baseline_info = get_baseline_compliance_info(rest_obj, name, attribute='Name')
    if not any(baseline_info):
        module.fail_json(msg=BASELINE_CHECK_MODE_NOCHANGE_MSG.format(name=name))
    current_payload = create_payload(module, rest_obj)
    current_payload['Id'] = baseline_info['Id']
    if module.params.get('new_name'):
        new_name = module.params.get('new_name')
        if name != new_name:
            baseline_info_new = get_baseline_compliance_info(rest_obj, new_name, attribute='Name')
            if any(baseline_info_new):
                module.fail_json(msg=BASELINE_CHECK_MODE_CHANGE_MSG.format(name=new_name))
        current_payload['Name'] = new_name
    required_attributes = ['Id', 'Name', 'Description', 'TemplateId', 'BaselineTargets']
    existing_payload = dict([(key, val) for key, val in baseline_info.items() if key in required_attributes and val])
    if existing_payload.get('BaselineTargets'):
        target = [{'Id': item['Id']} for item in existing_payload['BaselineTargets']]
        existing_payload['BaselineTargets'] = target
    idempotency_check_for_command_modify(existing_payload, current_payload, module)
    existing_payload.update(current_payload)
    baseline_update_uri = COMPLIANCE_BASELINE + '({baseline_id})'.format(baseline_id=existing_payload['Id'])
    resp = rest_obj.invoke_request('PUT', baseline_update_uri, data=existing_payload)
    data = resp.json_data
    compliance_id = data['Id']
    baseline_info = get_baseline_compliance_info(rest_obj, compliance_id)
    if module.params.get('job_wait'):
        job_failed, message = rest_obj.job_tracking(baseline_info['TaskId'], job_wait_sec=module.params['job_wait_timeout'], sleep_time=5)
        baseline_updated_info = get_baseline_compliance_info(rest_obj, compliance_id)
        if job_failed is True:
            module.fail_json(msg=message, compliance_status=baseline_updated_info, changed=False)
        elif 'successfully' in message:
            module.exit_json(msg=MODIFY_MSG, compliance_status=baseline_updated_info, changed=True)
        else:
            module.exit_json(msg=message, compliance_status=baseline_updated_info, changed=False)
    else:
        module.exit_json(msg=TASK_PROGRESS_MSG, compliance_status=baseline_info, changed=True)