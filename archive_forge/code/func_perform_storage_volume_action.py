from __future__ import (absolute_import, division, print_function)
import json
import copy
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import MANAGER_JOB_ID_URI, wait_for_redfish_reboot_job, \
def perform_storage_volume_action(method, uri, session_obj, action, payload=None):
    """
    common request call for raid creation update delete and initialization
    """
    try:
        resp = session_obj.invoke_request(method, uri, data=payload)
        task_uri = resp.headers['Location']
        return get_success_message(action, task_uri)
    except (HTTPError, URLError, SSLValidationError, ConnectionError, TypeError, ValueError) as err:
        raise err