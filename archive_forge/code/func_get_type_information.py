from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def get_type_information(rest_obj, uri):
    """
    rest_obj: Object containing information about connection to device.
    return: dict with information retrieved from URI.
    """
    type_info_dict = {}
    resp = rest_obj.invoke_request('GET', uri)
    if resp.status_code == 200:
        type_info = resp.json_data.get('value') if isinstance(resp.json_data.get('value'), list) else [resp.json_data]
        for item in type_info:
            item = clean_data(item)
            type_info_dict[item['Id']] = item
    return type_info_dict