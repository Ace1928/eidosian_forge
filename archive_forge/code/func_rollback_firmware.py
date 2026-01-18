from __future__ import (absolute_import, division, print_function)
import json
import re
import time
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params, \
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import wait_for_redfish_reboot_job, \
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def rollback_firmware(redfish_obj, module, preview_uri, reboot_uri, update_uri):
    current_job_status, failed_cnt, resetting = ([], 0, False)
    job_ids = simple_update(redfish_obj, preview_uri, update_uri)
    if module.params['reboot'] and preview_uri:
        payload = {'ResetType': 'ForceRestart'}
        job_resp_status, reset_status, reset_fail = wait_for_redfish_reboot_job(redfish_obj, SYSTEM_RESOURCE_ID, payload=payload)
        if reset_status and job_resp_status:
            job_uri = MANAGER_JOB_ID_URI.format(job_resp_status['Id'])
            job_resp, job_msg = wait_for_redfish_job_complete(redfish_obj, job_uri)
            job_status = job_resp.json_data
            if job_status['JobState'] != 'RebootCompleted':
                if job_msg:
                    module.fail_json(msg=JOB_WAIT_MSG.format(module.params['reboot_timeout']))
                else:
                    module.fail_json(msg=REBOOT_FAIL)
        elif not reset_status and reset_fail:
            module.fail_json(msg=reset_fail)
        current_job_status, failed = get_job_status(redfish_obj, module, job_ids, job_wait=True)
        failed_cnt += failed
    if not module.params['reboot'] and preview_uri:
        current_job_status, failed = get_job_status(redfish_obj, module, job_ids, job_wait=False)
        failed_cnt += failed
    if reboot_uri:
        job_ids = simple_update(redfish_obj, reboot_uri, update_uri)
        track, resetting, js_job_msg = wait_for_redfish_idrac_reset(module, redfish_obj, 900)
        if not track and resetting:
            reboot_job_status, failed = get_job_status(redfish_obj, module, job_ids, job_wait=True)
            current_job_status.extend(reboot_job_status)
            failed_cnt += failed
    return (current_job_status, failed_cnt, resetting)