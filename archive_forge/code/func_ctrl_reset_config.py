from __future__ import (absolute_import, division, print_function)
import json
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import wait_for_job_completion, strip_substr_dict
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def ctrl_reset_config(module, redfish_obj):
    resp, job_uri, job_id = (None, None, None)
    controller_id = module.params.get('controller_id')
    check_id_exists(module, redfish_obj, 'controller_id', controller_id, CONTROLLER_URI)
    member_resp = redfish_obj.invoke_request('GET', VOLUME_URI.format(system_id=SYSTEM_ID, controller_id=controller_id))
    members = member_resp.json_data.get('Members')
    if module.check_mode and members:
        module.exit_json(msg=CHANGES_FOUND, changed=True)
    elif module.check_mode and (not members) or (not module.check_mode and (not members)):
        module.exit_json(msg=NO_CHANGES_FOUND)
    else:
        resp = redfish_obj.invoke_request('POST', RAID_ACTION_URI.format(system_id=SYSTEM_ID, action=module.params['command']), data={'TargetFQDD': controller_id})
        job_uri = resp.headers.get('Location')
        job_id = job_uri.split('/')[-1]
    return (resp, job_uri, job_id)