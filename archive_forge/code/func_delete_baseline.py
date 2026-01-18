from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.common.dict_transformations import recursive_diff
def delete_baseline(module, rest_obj, baseline_list):
    delete_ids = []
    d = baseline_list[0]
    if d['TaskStatusId'] == 2050:
        module.fail_json(msg=BASELINE_JOB_RUNNING.format(name=d['Name'], id=d['Id']), job_id=d['TaskId'])
    delete_ids.append(d['Id'])
    delete_payload = {'BaselineIds': delete_ids}
    if module.check_mode:
        module.exit_json(msg=CHANGES_FOUND, changed=True)
    rest_obj.invoke_request('POST', DELETE_BASELINE_URI, data=delete_payload)
    module.exit_json(msg=BASELINE_DEL_SUCCESS, changed=True, baseline_id=delete_ids[0])