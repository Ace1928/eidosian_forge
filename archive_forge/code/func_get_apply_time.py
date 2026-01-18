from __future__ import (absolute_import, division, print_function)
import json
import copy
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import MANAGER_JOB_ID_URI, wait_for_redfish_reboot_job, \
def get_apply_time(module, session_obj, controller_id):
    """
    gets the apply time from user if given otherwise fetches from server
    """
    apply_time = module.params.get('apply_time')
    try:
        uri = APPLY_TIME_INFO_API.format(storage_base_uri=storage_collection_map['storage_base_uri'], controller_id=controller_id)
        resp = session_obj.invoke_request('GET', uri)
        supported_apply_time_values = resp.json_data['@Redfish.OperationApplyTimeSupport']['SupportedValues']
        if apply_time:
            if apply_time not in supported_apply_time_values:
                module.exit_json(msg=APPLY_TIME_NOT_SUPPORTED_MSG.format(apply_time=apply_time, supported_apply_time_values=supported_apply_time_values), failed=True)
        else:
            apply_time = supported_apply_time_values[0]
        return apply_time
    except (HTTPError, URLError, SSLValidationError, ConnectionError, TypeError, ValueError) as err:
        raise err