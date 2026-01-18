from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
def get_device_capabilities(devices_list, identifier):
    if identifier == 'device_ids':
        available_ids_capability_map = dict([(item['Id'], item.get('DeviceCapabilities', [])) for item in devices_list])
    else:
        available_ids_capability_map = dict([(item['Identifier'], item.get('DeviceCapabilities', [])) for item in devices_list])
    capable_devices = []
    noncapable_devices = []
    for key, val in available_ids_capability_map.items():
        if 33 in val:
            capable_devices.append(key)
        else:
            noncapable_devices.append(key)
    return {'capable': capable_devices, 'non_capable': noncapable_devices}