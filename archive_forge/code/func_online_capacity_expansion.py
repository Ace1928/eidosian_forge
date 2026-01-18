from __future__ import (absolute_import, division, print_function)
import json
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import wait_for_job_completion, strip_substr_dict
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def online_capacity_expansion(module, redfish_obj):
    payload = None
    volume_id = module.params.get('volume_id')
    target = module.params.get('target')
    size = module.params.get('size')
    if not isinstance(volume_id, list):
        volume_id = [volume_id]
    if len(volume_id) != 1:
        module.exit_json(msg=TARGET_ERR_MSG.format('virtual drive'), failed=True)
    controller_id = volume_id[0].split(':')[-1]
    volume_uri = VOLUME_URI + '/{volume_id}'
    try:
        volume_resp = redfish_obj.invoke_request('GET', volume_uri.format(system_id=SYSTEM_ID, controller_id=controller_id, volume_id=volume_id[0]))
    except HTTPError:
        module.exit_json(msg=VD_ERROR_MSG.format(volume_id[0]), failed=True)
    try:
        raid_type = volume_resp.json_data.get('RAIDType')
        if raid_type in ['RAID50', 'RAID60']:
            module.exit_json(msg=OCE_RAID_TYPE_ERR.format(raid_type), failed=True)
        if target is not None:
            if not target:
                module.exit_json(msg=OCE_TARGET_EMPTY, failed=True)
            if raid_type == 'RAID1':
                module.fail_json(msg=OCE_TARGET_RAID1_ERR)
            current_pd = []
            links = volume_resp.json_data.get('Links')
            if links:
                for disk in volume_resp.json_data.get('Links').get('Drives'):
                    drive = disk['@odata.id'].split('/')[-1]
                    current_pd.append(drive)
            drives_to_add = [each_drive for each_drive in target if each_drive not in current_pd]
            if module.check_mode and drives_to_add and (len(drives_to_add) % OCE_MIN_PD_RAID_MAPPING[raid_type] == 0):
                module.exit_json(msg=CHANGES_FOUND, changed=True)
            elif len(drives_to_add) == 0 or len(drives_to_add) % OCE_MIN_PD_RAID_MAPPING[raid_type] != 0:
                module.exit_json(msg=NO_CHANGES_FOUND)
            payload = {'TargetFQDD': volume_id[0], 'PDArray': drives_to_add}
        elif size:
            vd_size = volume_resp.json_data.get('CapacityBytes')
            vd_size_MB = vd_size // (1024 * 1024)
            if size - vd_size_MB < 100:
                module.exit_json(msg=OCE_SIZE_100MB.format(vd_size_MB), failed=True)
            payload = {'TargetFQDD': volume_id[0], 'Size': size}
        resp = redfish_obj.invoke_request('POST', RAID_ACTION_URI.format(system_id=SYSTEM_ID, action='OnlineCapacityExpansion'), data=payload)
        job_uri = resp.headers.get('Location')
        job_id = job_uri.split('/')[-1]
        return (resp, job_uri, job_id)
    except HTTPError as err:
        err = json.load(err).get('error').get('@Message.ExtendedInfo', [{}])[0].get('Message')
        module.exit_json(msg=err, failed=True)