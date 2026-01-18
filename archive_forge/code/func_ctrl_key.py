from __future__ import (absolute_import, division, print_function)
import json
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import wait_for_job_completion, strip_substr_dict
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def ctrl_key(module, redfish_obj):
    resp, job_uri, job_id, payload = (None, None, None, {})
    controller_id = module.params.get('controller_id')
    command, mode = (module.params['command'], module.params['mode'])
    key, key_id = (module.params.get('key'), module.params.get('key_id'))
    check_id_exists(module, redfish_obj, 'controller_id', controller_id, CONTROLLER_URI)
    ctrl_resp = redfish_obj.invoke_request('GET', CONTROLLER_URI.format(system_id=SYSTEM_ID, controller_id=controller_id))
    security_status = ctrl_resp.json_data.get('SecurityStatus')
    if security_status == 'EncryptionNotCapable':
        module.fail_json(msg=ENCRYPT_ERR_MSG.format(controller_id))
    ctrl_key_id = ctrl_resp.json_data.get('KeyID')
    if command == 'SetControllerKey':
        if module.check_mode and ctrl_key_id is None:
            module.exit_json(msg=CHANGES_FOUND, changed=True)
        elif module.check_mode and ctrl_key_id is not None or (not module.check_mode and ctrl_key_id is not None):
            module.exit_json(msg=NO_CHANGES_FOUND)
        payload = {'TargetFQDD': controller_id, 'Key': key, 'Keyid': key_id}
    elif command == 'ReKey':
        if module.check_mode:
            module.exit_json(msg=CHANGES_FOUND, changed=True)
        if mode == 'LKM':
            payload = {'TargetFQDD': controller_id, 'Mode': mode, 'NewKey': key, 'Keyid': key_id, 'OldKey': module.params.get('old_key')}
        else:
            payload = {'TargetFQDD': controller_id, 'Mode': mode}
    elif command == 'RemoveControllerKey':
        if module.check_mode and ctrl_key_id is not None:
            module.exit_json(msg=CHANGES_FOUND, changed=True)
        elif module.check_mode and ctrl_key_id is None or (not module.check_mode and ctrl_key_id is None):
            module.exit_json(msg=NO_CHANGES_FOUND)
        payload = {'TargetFQDD': controller_id}
    elif command == 'EnableControllerEncryption':
        if module.check_mode and (not security_status == 'SecurityKeyAssigned'):
            module.exit_json(msg=CHANGES_FOUND, changed=True)
        elif module.check_mode and security_status == 'SecurityKeyAssigned' or (not module.check_mode and security_status == 'SecurityKeyAssigned'):
            module.exit_json(msg=NO_CHANGES_FOUND)
        payload = {'TargetFQDD': controller_id, 'Mode': mode}
        if mode == 'LKM':
            payload['Key'] = key
            payload['Keyid'] = key_id
    resp = redfish_obj.invoke_request('POST', RAID_ACTION_URI.format(system_id=SYSTEM_ID, action=command), data=payload)
    job_uri = resp.headers.get('Location')
    job_id = job_uri.split('/')[-1]
    return (resp, job_uri, job_id)