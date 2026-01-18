from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
def remediate_baseline(module, rest_obj):
    noncomplaint_devices, baseline_info = validate_remediate_idempotency(module, rest_obj)
    remediate_payload = create_remediate_payload(noncomplaint_devices, baseline_info, rest_obj)
    resp = rest_obj.invoke_request('POST', REMEDIATE_BASELINE, data=remediate_payload)
    job_id = resp.json_data
    if module.params.get('job_wait'):
        job_failed, message = rest_obj.job_tracking(job_id, job_wait_sec=module.params['job_wait_timeout'])
        if job_failed is True:
            module.fail_json(msg=message, job_id=job_id, changed=False)
        elif 'successfully' in message:
            module.exit_json(msg=REMEDIATE_MSG, job_id=job_id, changed=True)
        else:
            module.exit_json(msg=message, job_id=job_id, changed=False)
    else:
        module.exit_json(msg=TASK_PROGRESS_MSG, job_id=job_id, changed=True)