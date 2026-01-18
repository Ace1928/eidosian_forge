from __future__ import (absolute_import, division, print_function)
import json
import copy
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import MANAGER_JOB_ID_URI, wait_for_redfish_reboot_job, \
def check_apply_time_supported_and_reboot_required(module, session_obj, controller_id):
    """
    checks whether the apply time is supported and reboot operation is required or not.
    """
    apply_time = get_apply_time(module, session_obj, controller_id)
    reboot_server = module.params.get('reboot_server')
    if reboot_server and apply_time == 'OnReset':
        return True
    return False