from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (strip_substr_dict, idrac_system_reset,
from ansible.module_utils.basic import AnsibleModule
def get_existing_boot_options(idrac, res_id):
    resp = idrac.invoke_request(BOOT_OPTIONS_URI.format(res_id), 'GET')
    resp_data = strip_substr_dict(resp.json_data)
    strip_members = []
    for each in resp_data['Members']:
        strip_members.append(strip_substr_dict(each))
    resp_data['Members'] = strip_members
    return resp_data