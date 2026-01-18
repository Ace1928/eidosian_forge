from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import apply_diff_key, job_tracking
def get_group_details(rest_obj, module):
    group_name_list = module.params.get('device_group_names')
    device_ids = []
    for group_name in group_name_list:
        group = get_group(rest_obj, module, group_name)
        group_uri = GROUP_URI + '({0})/Devices'.format(group['Id'])
        group_device_list = get_group_devices_all(rest_obj, group_uri)
        device_ids.extend([dev['Id'] for dev in group_device_list])
    return device_ids