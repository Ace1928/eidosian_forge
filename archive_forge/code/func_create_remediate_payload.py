from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
def create_remediate_payload(noncomplaint_devices, baseline_info, rest_obj):
    ome_version = get_ome_version(rest_obj)
    payload = {'Id': baseline_info['Id'], 'Schedule': {'RunNow': True, 'RunLater': False}}
    if LooseVersion(ome_version) >= '3.5':
        payload['DeviceIds'] = noncomplaint_devices
    else:
        payload['TargetIds'] = noncomplaint_devices
    return payload