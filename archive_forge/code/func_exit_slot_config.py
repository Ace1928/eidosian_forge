from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def exit_slot_config(module, rest_obj, failed_jobs, invalid_jobs, slot_data):
    failed_jobs.update(invalid_jobs)
    if failed_jobs:
        f = len(failed_jobs)
        s = len(slot_data)
        slot_info = get_formatted_slotlist(slot_data)
        failed_jobs_list = get_formatted_slotlist(failed_jobs)
        module.fail_json(msg=FAILED_MSG.format(f, s + f), slot_info=slot_info, rename_failed_slots=failed_jobs_list)
    if slot_data:
        job_failed_list = []
        try:
            rfrsh_job_list = trigger_refresh_inventory(rest_obj, slot_data)
            for job in rfrsh_job_list:
                job_failed, job_message = rest_obj.job_tracking(job, job_wait_sec=JOB_TIMEOUT, sleep_time=JOB_INTERVAL)
                job_failed_list.append(job_failed)
            all_dv_rfrsh = trigger_all_inventory_task(rest_obj)
            job_failed, job_message = rest_obj.job_tracking(all_dv_rfrsh, job_wait_sec=JOB_TIMEOUT, sleep_time=JOB_INTERVAL)
            job_failed_list.append(job_failed)
        except Exception:
            job_failed_list = [True]
        if any(job_failed_list) is True:
            slot_info = get_formatted_slotlist(slot_data)
            failed_jobs_list = get_formatted_slotlist(failed_jobs)
            module.exit_json(changed=True, msg=SUCCESS_REFRESH_MSG, slot_info=slot_info, rename_failed_slots=failed_jobs_list)
    slot_info = get_formatted_slotlist(slot_data)
    module.exit_json(changed=True, msg=SUCCESS_MSG, slot_info=slot_info, rename_failed_slots=list(failed_jobs.values()))