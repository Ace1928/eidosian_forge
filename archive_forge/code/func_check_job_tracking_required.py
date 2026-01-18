from __future__ import (absolute_import, division, print_function)
import json
import copy
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import MANAGER_JOB_ID_URI, wait_for_redfish_reboot_job, \
def check_job_tracking_required(module, session_obj, reboot_required, controller_id):
    job_wait = module.params.get('job_wait')
    apply_time = None
    if controller_id:
        apply_time = get_apply_time(module, session_obj, controller_id)
    if job_wait:
        if apply_time == 'OnReset' and (not reboot_required):
            return False
        return True
    return False