from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def get_job_states(module, rest_obj, slot_data):
    job_dict = dict([(slot['JobId'], k) for k, slot in slot_data.items() if slot['JobId']])
    query_params = {'$filter': 'JobType/Id eq 3'}
    count = JOB_TIMEOUT // SETTLING_TIME
    job_incomplete = [2050, 2030, 2040, 2080]
    while count > 0 and job_dict:
        try:
            job_resp = rest_obj.invoke_request('GET', JOB_URI, query_param=query_params)
            jobs = job_resp.json_data.get('value')
        except HTTPError:
            count = count - 50
            time.sleep(SETTLING_TIME)
            continue
        job_over = []
        for job in jobs:
            id = job.get('Id')
            if id in job_dict:
                lrs = job.get('LastRunStatus')
                slot = slot_data[job_dict[id]]
                if lrs.get('Id') in job_incomplete:
                    job_over.append(False)
                elif lrs.get('Id') == 2060:
                    job_over.append(True)
                    slot['SlotName'] = slot.pop('new_name')
                    job_dict.pop(id)
                else:
                    slot['JobStatus'] = lrs.get('Name')
                    job_over.append(True)
        if all(job_over) or not job_dict:
            break
        count = count - 1
        time.sleep(SETTLING_TIME)
    failed_jobs = dict([(k, slot_data.pop(k)) for k in job_dict.values()])
    return failed_jobs