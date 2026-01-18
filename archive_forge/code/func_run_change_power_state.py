from __future__ import (absolute_import, division, print_function)
import json
import re
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
def run_change_power_state(redfish_session_obj, module):
    """
    Apply reset type to system
    Keyword arguments:
    redfish_session_obj  -- session handle
    module -- Ansible module obj
    """
    apply_reset_type = module.params['reset_type']
    fetch_power_uri_resource(module, redfish_session_obj)
    is_valid_reset_type(apply_reset_type, powerstate_map['allowable_enums'], module)
    current_power_state = powerstate_map['current_state']
    reset_flag = is_change_applicable_for_power_state(current_power_state, apply_reset_type)
    if module.check_mode is True:
        if reset_flag is True:
            module.exit_json(msg='Changes found to be applied.', changed=True)
        else:
            module.exit_json(msg='No Changes found to be applied.', changed=False)
    if reset_flag is True:
        payload = {'ResetType': apply_reset_type}
        power_uri = powerstate_map['power_uri']
        reset_resp = redfish_session_obj.invoke_request('POST', power_uri, data=payload)
        if reset_resp.success:
            module.exit_json(msg="Successfully performed the reset type operation '{0}'.".format(apply_reset_type), changed=True)
        else:
            module.exit_json(msg="Unable to perform the reset type operation '{0}'.".format(apply_reset_type), changed=False)
    else:
        module.exit_json(msg='The device is already powered {0}.'.format(current_power_state.lower()), changed=False)