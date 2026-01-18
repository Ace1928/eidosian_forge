from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def get_protocol_device_map(rest_obj):
    prot_dev_map = {}
    dev_id_map = {}
    resp = rest_obj.invoke_request('GET', PROTOCOL_DEVICE)
    prot_dev = resp.json_data.get('value')
    for item in prot_dev:
        dname = item['DeviceTypeName']
        dlist = prot_dev_map.get(dname, [])
        dlist.append(item['ProtocolName'])
        prot_dev_map[dname] = dlist
        dev_id_map[dname] = item['DeviceTypeId']
        if dname == 'DELL STORAGE':
            prot_dev_map['STORAGE'] = dlist
            dev_id_map['STORAGE'] = item['DeviceTypeId']
    return (prot_dev_map, dev_id_map)