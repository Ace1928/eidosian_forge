from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import idrac_redfish_job_tracking, \
def check_scheduled_bios_job(redfish_obj):
    job_resp = redfish_obj.invoke_request(iDRAC_JOBS_EXP, 'GET')
    job_list = job_resp.json_data.get('Members', [])
    sch_jb = None
    jb_state = 'Unknown'
    for jb in job_list:
        if jb.get('JobType') == 'BIOSConfiguration' and jb.get('JobState') in ['Scheduled', 'Running', 'Starting']:
            sch_jb = jb['Id']
            jb_state = jb.get('JobState')
            break
    return (sch_jb, jb_state)