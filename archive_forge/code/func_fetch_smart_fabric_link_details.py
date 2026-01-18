from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def fetch_smart_fabric_link_details(module, rest_obj, fabric_details_dict):
    info_dict = {'Switches': 'Switches@odata.navigationLink', 'Servers': 'Servers@odata.navigationLink', 'ISLLinks': 'ISLLinks@odata.navigationLink', 'Uplinks': 'Uplinks@odata.navigationLink', 'Multicast': None, 'FabricDesign': None}
    info_list = ['Multicast', 'FabricDesign']
    try:
        for key in info_dict:
            link = info_dict[key]
            if key in info_list:
                fabric_info_dict = fabric_details_dict[key]['@odata.id']
                uri = fabric_info_dict.strip('/api')
                response = rest_obj.invoke_request('GET', uri)
                if response.json_data:
                    details = [response.json_data]
            else:
                fabric_info_dict = fabric_details_dict.get(link)
                uri = fabric_info_dict.strip('/api')
                response = rest_obj.invoke_request('GET', uri)
                if response.json_data:
                    details = response.json_data.get('value')
            for item in details:
                item = strip_substr_dict(item)
                item = clean_data(item)
                fabric_details_dict[key] = details
    except HTTPError:
        module.exit_json(msg=UNSUCCESS_MSG, failed=True)
    return fabric_details_dict