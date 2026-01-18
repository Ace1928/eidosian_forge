from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.common.dict_transformations import recursive_diff
def migrate_profile(module, rest_obj):
    mparam = module.params
    payload = {}
    payload['ForceMigrate'] = mparam.get('force')
    target = get_target_details(module, rest_obj)
    if not isinstance(target, dict):
        module.fail_json(msg=target)
    payload['TargetId'] = target['Id']
    prof = get_profile(rest_obj, module)
    if prof:
        if target['Id'] == prof['TargetId']:
            module.exit_json(msg=NO_CHANGES_MSG)
        try:
            resp = rest_obj.invoke_request('POST', PROFILE_ACTION.format(action='GetInvalidTargetsForAssignProfile'), data={'Id': prof['Id']})
            if target['Id'] in list(resp.json_data):
                module.fail_json(msg='The target device is invalid for the given profile.')
        except HTTPError:
            resp = None
        if prof['ProfileState'] == 4:
            payload['ProfileId'] = prof['Id']
            if module.check_mode:
                module.exit_json(msg=CHANGES_MSG, changed=True)
            resp = rest_obj.invoke_request('POST', PROFILE_ACTION.format(action='MigrateProfile'), data=payload)
            msg = 'Successfully applied the migrate operation.'
            res_dict = {'msg': msg, 'changed': True}
            try:
                time.sleep(5)
                res_prof = get_profile(rest_obj, module)
                if res_prof.get('DeploymentTaskId'):
                    res_dict['job_id'] = res_prof.get('DeploymentTaskId')
                    res_dict['msg'] = 'Successfully triggered the job for the migrate operation.'
            except HTTPError:
                res_dict['msg'] = 'Successfully applied the migrate operation. Failed to fetch job details.'
            module.exit_json(**res_dict)
        else:
            module.fail_json(msg='Profile needs to be in a deployed state for a migrate operation.')
    else:
        module.fail_json(msg=PROFILE_NOT_FOUND.format(name=mparam.get('name')))