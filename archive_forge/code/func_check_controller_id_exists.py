from __future__ import (absolute_import, division, print_function)
import json
import copy
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import MANAGER_JOB_ID_URI, wait_for_redfish_reboot_job, \
def check_controller_id_exists(module, session_obj):
    """
    Controller availability Validation
    """
    specified_controller_id = module.params.get('controller_id')
    uri = CONTROLLER_URI.format(storage_base_uri=storage_collection_map['storage_base_uri'], controller_id=specified_controller_id)
    err_message = CONTROLLER_NOT_EXIST_ERROR.format(controller_id=specified_controller_id)
    resp = check_specified_identifier_exists_in_the_system(module, session_obj, uri, err_message)
    if resp.success:
        return check_physical_disk_exists(module, resp.json_data['Drives'])
    else:
        module.fail_json(msg='Failed to retrieve the details of the specified Controller Id {0}.'.format(specified_controller_id))