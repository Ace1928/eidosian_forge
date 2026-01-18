from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import apply_diff_key, job_tracking
def exit_module(rest_obj, module, response, time_out=False):
    password_no_log(module.params.get('attributes'))
    resp = None
    changed_flag = True
    command = module.params.get('command')
    result = {}
    if command in ['create', 'modify', 'deploy', 'import', 'clone']:
        result['return_id'] = response.json_data
        resp = result['return_id']
        if command == 'deploy':
            if time_out:
                command = 'timed_out'
                changed_flag = False
            elif not result['return_id']:
                result['failed'] = True
                command = 'deploy_fail'
                changed_flag = False
            elif module.params['job_wait']:
                command = 'deploy_when_job_wait_true'
            else:
                command = 'deploy_when_job_wait_false'
        elif command == 'create':
            if time_out:
                resp = get_job_id(rest_obj, resp)
                command = 'timed_out'
                changed_flag = False
            elif module.params['job_wait']:
                command = 'create_when_job_wait_true'
            else:
                time.sleep(5)
                resp = get_job_id(rest_obj, resp)
                command = 'create_when_job_wait_false'
    if command == 'export':
        changed_flag = False
        result = response.json_data
    message = MSG_DICT.get(command).format(resp)
    module.exit_json(msg=message, changed=changed_flag, **result)