from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import remove_key
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def get_all_alert_policy_info(self, rest_obj) -> dict:
    resp = rest_obj.invoke_request('GET', ALERT_POLICY_URI)
    value = resp.json_data['value']
    output_all = {'msg': MODULE_SUCCESS_MESSAGE_ALL, 'value': remove_key(value)}
    return output_all