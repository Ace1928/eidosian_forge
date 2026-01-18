from __future__ import (absolute_import, division, print_function)
import json
import re
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_manager_res_id
from ansible.module_utils.basic import AnsibleModule
def get_response_attr(idrac, idrac_id, attr, uri_dict):
    response_attr = {}
    diff = 0
    response = idrac.invoke_request(uri_dict.get(idrac_id), 'GET')
    for k in attr.keys():
        if response.json_data[ATTR].get(k) != attr.get(k):
            response_attr[k] = attr.get(k)
            diff += 1
    return (diff, response_attr)