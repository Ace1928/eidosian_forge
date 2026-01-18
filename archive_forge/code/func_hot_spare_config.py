from __future__ import (absolute_import, division, print_function)
import json
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import wait_for_job_completion, strip_substr_dict
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def hot_spare_config(module, redfish_obj):
    target, command = (module.params.get('target'), module.params['command'])
    resp, job_uri, job_id = (None, None, None)
    volume = module.params.get('volume_id')
    controller_id = target[0].split(':')[-1]
    drive_id = target[0]
    try:
        pd_resp = redfish_obj.invoke_request('GET', PD_URI.format(controller_id=controller_id, drive_id=drive_id))
    except HTTPError:
        module.fail_json(msg=PD_ERROR_MSG.format(drive_id))
    else:
        hot_spare = pd_resp.json_data.get('HotspareType')
        if module.check_mode and hot_spare == 'None' and (command == 'AssignSpare') or (module.check_mode and (not hot_spare == 'None') and (command == 'UnassignSpare')):
            module.exit_json(msg=CHANGES_FOUND, changed=True)
        elif module.check_mode and hot_spare in ['Dedicated', 'Global'] and (command == 'AssignSpare') or (not module.check_mode and hot_spare in ['Dedicated', 'Global'] and (command == 'AssignSpare')) or (module.check_mode and hot_spare == 'None' and (command == 'UnassignSpare')) or (not module.check_mode and hot_spare == 'None' and (command == 'UnassignSpare')):
            module.exit_json(msg=NO_CHANGES_FOUND)
        else:
            payload = {'TargetFQDD': drive_id}
            if volume is not None and command == 'AssignSpare':
                payload['VirtualDiskArray'] = volume
            resp = redfish_obj.invoke_request('POST', RAID_ACTION_URI.format(system_id=SYSTEM_ID, action=command), data=payload)
            job_uri = resp.headers.get('Location')
            job_id = job_uri.split('/')[-1]
    return (resp, job_uri, job_id)