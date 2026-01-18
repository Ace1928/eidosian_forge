from __future__ import (absolute_import, division, print_function)
import json
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import wait_for_job_completion, strip_substr_dict
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def convert_raid_status(module, redfish_obj):
    resp, job_uri, job_id = (None, None, None)
    command, target = (module.params['command'], module.params.get('target'))
    ctrl, pd_ready_state = (None, [])
    try:
        for ctrl in target:
            controller_id = ctrl.split(':')[-1]
            pd_resp = redfish_obj.invoke_request('GET', PD_URI.format(controller_id=controller_id, drive_id=ctrl))
            raid_status = pd_resp.json_data['Oem']['Dell']['DellPhysicalDisk']['RaidStatus']
            pd_ready_state.append(raid_status)
    except HTTPError:
        module.fail_json(msg=PD_ERROR_MSG.format(ctrl))
    else:
        if command == 'ConvertToRAID' and module.check_mode and (0 < pd_ready_state.count('NonRAID')) or (command == 'ConvertToNonRAID' and module.check_mode and (0 < pd_ready_state.count('Ready'))):
            module.exit_json(msg=CHANGES_FOUND, changed=True)
        elif command == 'ConvertToRAID' and module.check_mode and (len(pd_ready_state) == pd_ready_state.count('Ready')) or (command == 'ConvertToRAID' and (not module.check_mode) and (len(pd_ready_state) == pd_ready_state.count('Ready'))) or (command == 'ConvertToNonRAID' and module.check_mode and (len(pd_ready_state) == pd_ready_state.count('NonRAID'))) or (command == 'ConvertToNonRAID' and (not module.check_mode) and (len(pd_ready_state) == pd_ready_state.count('NonRAID'))):
            module.exit_json(msg=NO_CHANGES_FOUND)
        else:
            resp = redfish_obj.invoke_request('POST', RAID_ACTION_URI.format(system_id=SYSTEM_ID, action=command), data={'PDArray': target})
            job_uri = resp.headers.get('Location')
            job_id = job_uri.split('/')[-1]
    return (resp, job_uri, job_id)