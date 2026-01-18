from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict, job_tracking
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import CHANGES_MSG, NO_CHANGES_MSG
def perform_device_tasks(module, rest_obj, valid_ids):
    task = module.params.get('device_action')
    payload, method, uri = get_payload_method(task, valid_ids)
    update_common_job(module, payload, task, valid_ids)
    job = check_similar_job(rest_obj, payload)
    if not job:
        formalize_job_payload(payload)
        if module.check_mode:
            module.exit_json(msg=CHANGES_MSG, changed=True)
        resp = rest_obj.invoke_request('POST', JOBS_URI, data=payload, api_timeout=60)
        job_wait(module, rest_obj, resp.json_data)
    else:
        if module.params.get('job_schedule') == 'startnow' and job['LastRunStatus']['Id'] != 2050:
            if module.check_mode:
                module.exit_json(msg=CHANGES_MSG, changed=True)
            resp = rest_obj.invoke_request('POST', RUN_JOB_URI, data={'JobIds': [job['Id']]})
            job_wait(module, rest_obj, job)
        module.exit_json(msg=NO_CHANGES_MSG, job=strip_substr_dict(job))