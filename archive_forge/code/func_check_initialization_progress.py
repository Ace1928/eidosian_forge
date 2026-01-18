from __future__ import (absolute_import, division, print_function)
import json
import copy
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import MANAGER_JOB_ID_URI, wait_for_redfish_reboot_job, \
def check_initialization_progress(module, session_obj, volume_id):
    """
    validation check if any operation is running in specified volume id.
    """
    operations = []
    resp = check_volume_id_exists(module, session_obj, volume_id)
    if resp.success:
        operations = resp.json_data['Operations']
    return operations