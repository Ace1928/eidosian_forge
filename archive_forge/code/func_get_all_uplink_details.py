from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def get_all_uplink_details(module, rest_obj):
    resp = []
    try:
        fabric_det = rest_obj.invoke_request('GET', FABRIC_URI)
        fabric_resp = fabric_det.json_data.get('value')
        for each in fabric_resp:
            if each.get('Uplinks@odata.navigationLink'):
                uplink_det = each.get('Uplinks@odata.navigationLink')
                uplink = uplink_det[5:] + '?$expand=Networks,Ports'
                uplink_details = rest_obj.invoke_request('GET', uplink)
                for val in uplink_details.json_data.get('value'):
                    resp.append(val)
    except HTTPError:
        module.exit_json(msg=UNSUCCESS_MSG, failed=True)
    return resp