from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def start_slot_name_jobs(rest_obj, slot_data):
    slot_type = {'2000': 'Sled Slot', '4000': 'IO Module Slot', '2100': 'Storage Sled'}
    failed_jobs = {}
    job_description = SLOT_JOB_DESC
    job_type = {'Id': 3, 'Name': 'DeviceAction_Task'}
    for k, slot in slot_data.items():
        job_params, target_param = ([{'Key': 'operationName', 'Value': 'UPDATE_SLOT_DATA'}], [])
        num = slot.get('SlotNumber')
        type_id = str(slot.get('SlotType'))
        job_name = 'Rename {0} {1}'.format(slot_type.get(type_id, 'Slot'), num)
        target_param.append({'Id': int(slot.get('ChassisId')), 'Data': '', 'TargetType': {'Id': 1000, 'Name': 'DEVICE'}})
        slot_config = '{0}|{1}|{2}'.format(num, type_id, slot.get('new_name'))
        job_params.append({'Key': 'slotConfig', 'Value': slot_config})
        try:
            job_resp = rest_obj.job_submission(job_name, job_description, target_param, job_params, job_type)
            slot['JobId'] = job_resp.json_data.get('Id', 0)
            time.sleep(SETTLING_TIME)
        except HTTPError as err:
            slot['JobId'] = 0
            slot['JobStatus'] = str(err)
            failed_jobs[k] = slot
    [slot_data.pop(key) for key in failed_jobs.keys()]
    return failed_jobs