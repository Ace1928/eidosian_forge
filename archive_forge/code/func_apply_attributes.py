from __future__ import (absolute_import, division, print_function)
import json
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import wait_for_job_completion, strip_substr_dict
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def apply_attributes(module, redfish_obj, pending, time_settings):
    payload = {'Oem': {'Dell': {'DellStorageController': pending}}}
    apply_time = module.params.get('apply_time')
    time_set = get_redfish_apply_time(module, redfish_obj, apply_time, time_settings)
    if time_set:
        payload['@Redfish.SettingsApplyTime'] = time_set
    try:
        resp = redfish_obj.invoke_request('PATCH', SETTINGS_URI.format(system_id=SYSTEM_ID, controller_id=module.params['controller_id']), data=payload)
        if resp.status_code == 202 and 'error' in resp.json_data:
            msg_err_id = resp.json_data.get('error').get('@Message.ExtendedInfo', [{}])[0].get('MessageId')
            if 'Created' not in msg_err_id:
                module.exit_json(msg=ERR_MSG, error_info=resp.json_data, failed=True)
    except HTTPError as err:
        err = json.load(err).get('error')
        module.exit_json(msg=ERR_MSG, error_info=err, failed=True)
    job_id = resp.headers['Location'].split('/')[-1]
    return (job_id, time_set)