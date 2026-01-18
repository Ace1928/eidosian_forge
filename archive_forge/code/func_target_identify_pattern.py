from __future__ import (absolute_import, division, print_function)
import json
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import wait_for_job_completion, strip_substr_dict
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def target_identify_pattern(module, redfish_obj):
    target, volume = (module.params.get('target'), module.params.get('volume_id'))
    command = module.params.get('command')
    payload = {'TargetFQDD': None}
    if target is not None and volume is None:
        payload = {'TargetFQDD': target[0]}
    elif volume is not None and target is None:
        payload = {'TargetFQDD': volume[0]}
    elif target is not None and volume is not None:
        payload = {'TargetFQDD': target[0]}
    if module.check_mode:
        module.exit_json(msg=CHANGES_FOUND, changed=True)
    resp = redfish_obj.invoke_request('POST', RAID_ACTION_URI.format(system_id=SYSTEM_ID, action=command), data=payload)
    return resp