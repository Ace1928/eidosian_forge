from __future__ import (absolute_import, division, print_function)
import json
import copy
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import MANAGER_JOB_ID_URI, wait_for_redfish_reboot_job, \
def perform_force_reboot(module, session_obj):
    payload = {'ResetType': 'ForceRestart'}
    job_resp_status, reset_status, reset_fail = wait_for_redfish_reboot_job(session_obj, SYSTEM_ID, payload=payload)
    if reset_status and job_resp_status:
        job_uri = MANAGER_JOB_ID_URI.format(job_resp_status['Id'])
        resp, msg = wait_for_job_completion(session_obj, job_uri, wait_timeout=module.params.get('job_wait_timeout'))
        if resp:
            job_data = strip_substr_dict(resp.json_data)
            if job_data['JobState'] == 'Failed':
                module.exit_json(msg=REBOOT_FAIL, job_status=job_data, failed=True)
        else:
            resp = session_obj.invoke_request('GET', job_uri)
            job_data = strip_substr_dict(resp.json_data)
            module.exit_json(msg=msg, job_status=job_data)