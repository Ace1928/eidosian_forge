from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import idrac_redfish_job_tracking, \
def reset_host(module, redfish_obj):
    reset_type = module.params.get('reset_type')
    p_state = 'On'
    ps = get_power_state(redfish_obj)
    on_state = ['On']
    if ps in on_state:
        p_state = 'GracefulShutdown'
        if 'force' in reset_type:
            p_state = 'ForceOff'
        p_act = power_act_host(redfish_obj, p_state)
        if not p_act:
            module.exit_json(failed=True, status_msg=HOST_RESTART_FAILED)
        state_achieved = track_power_state(redfish_obj, ['Off'])
        p_state = 'On'
        if not state_achieved:
            time.sleep(10)
            p_state = 'ForceRestart'
    p_act = power_act_host(redfish_obj, p_state)
    if not p_act:
        module.exit_json(failed=True, status_msg=HOST_RESTART_FAILED)
    state_achieved = track_power_state(redfish_obj, on_state)
    return state_achieved