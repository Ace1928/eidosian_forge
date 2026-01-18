from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
def create_baseline(module, rest_obj):
    """
    Create the compliance baseline.
    update the response by getting compliance info.
    Note: The response is updated from GET info reason many attribute values are gving null
    value. which can be retrieved by getting the created compliance info.
    """
    payload = create_payload(module, rest_obj)
    validate_create_baseline_idempotency(module, rest_obj)
    resp = rest_obj.invoke_request('POST', COMPLIANCE_BASELINE, data=payload)
    data = resp.json_data
    compliance_id = data['Id']
    baseline_info = get_baseline_compliance_info(rest_obj, compliance_id)
    if module.params.get('job_wait'):
        job_failed, message = rest_obj.job_tracking(baseline_info['TaskId'], job_wait_sec=module.params['job_wait_timeout'], sleep_time=5)
        baseline_updated_info = get_baseline_compliance_info(rest_obj, compliance_id)
        if job_failed is True:
            module.fail_json(msg=message, compliance_status=baseline_updated_info, changed=False)
        elif 'successfully' in message:
            module.exit_json(msg=CREATE_MSG, compliance_status=baseline_updated_info, changed=True)
        else:
            module.exit_json(msg=message, compliance_status=baseline_updated_info, changed=False)
    else:
        module.exit_json(msg=TASK_PROGRESS_MSG, compliance_status=baseline_info, changed=True)