from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (strip_substr_dict, idrac_system_reset,
from ansible.module_utils.basic import AnsibleModule
def apply_boot_settings(module, idrac, payload, res_id):
    job_data, job_wait = ({}, module.params['job_wait'])
    if module.params['reset_type'] == 'none':
        job_wait = False
    resp = idrac.invoke_request('{0}/{1}'.format(SYSTEM_URI, res_id), 'PATCH', data=payload)
    if resp.status_code == 200:
        reset, track_failed, reset_msg, reset_job_resp = system_reset(module, idrac, res_id)
        if reset_job_resp:
            job_data = reset_job_resp.json_data
        is_job, progress_job = get_scheduled_job(idrac)
        if is_job:
            if reset:
                job_resp, error_msg = wait_for_idrac_job_completion(idrac, JOB_URI_ID.format(progress_job[0]['Id']), job_wait=job_wait, wait_timeout=module.params['job_wait_timeout'])
                if error_msg:
                    module.fail_json(msg=error_msg)
                job_data = job_resp.json_data
            else:
                module.fail_json(msg=reset_msg)
    return job_data