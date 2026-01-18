from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (strip_substr_dict, idrac_system_reset,
from ansible.module_utils.basic import AnsibleModule
def get_scheduled_job(idrac, job_state=None):
    if job_state is None:
        job_state = ['Scheduled', 'New', 'Running']
    is_job, job_type_name, progress_job = (False, 'BIOSConfiguration', [])
    time.sleep(10)
    job_resp = idrac.invoke_request(JOB_URI, 'GET')
    job_resp_member = job_resp.json_data['Members']
    if job_resp_member:
        bios_config_job = list(filter(lambda d: d.get('JobType') in [job_type_name], job_resp_member))
        progress_job = list(filter(lambda d: d.get('JobState') in job_state, bios_config_job))
        if progress_job:
            is_job = True
    return (is_job, progress_job)