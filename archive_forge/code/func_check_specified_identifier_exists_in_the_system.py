from __future__ import (absolute_import, division, print_function)
import json
import copy
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import MANAGER_JOB_ID_URI, wait_for_redfish_reboot_job, \
def check_specified_identifier_exists_in_the_system(module, session_obj, uri, err_message):
    """
    common validation to check if , specified volume or controller id exist in the system or not
    """
    try:
        resp = session_obj.invoke_request('GET', uri)
        return resp
    except HTTPError as err:
        if err.code == 404:
            module.exit_json(msg=err_message, failed=True)
        raise err
    except (URLError, SSLValidationError, ConnectionError, TypeError, ValueError) as err:
        raise err