from __future__ import (absolute_import, division, print_function)
import json
import re
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def device_validation(module, rest_obj):
    device_lst, invalid_lst, other_types = ([], [], [])
    devices, tags = (module.params.get('device_ids'), module.params.get('device_service_tags'))
    all_device = rest_obj.get_all_report_details(DEVICE_URI)
    key = 'Id' if devices is not None else 'DeviceServiceTag'
    value = 'id' if key == 'Id' else 'service tag'
    req_device = devices if devices is not None else tags
    for each in req_device:
        device = list(filter(lambda d: d[key] in [each], all_device['report_list']))
        if device and device[0]['Type'] == 1000:
            device_lst.append(device[0]['Id'])
        elif device and (not device[0]['Type'] == 1000):
            other_types.append(str(each))
        else:
            invalid_lst.append(str(each))
    if invalid_lst:
        module.fail_json(msg="Unable to complete the operation because the entered target device {0}(s) '{1}' are invalid.".format(value, ','.join(set(invalid_lst))))
    if not device_lst and other_types:
        module.fail_json(msg="The requested device {0}(s) '{1}' are not applicable for export log.".format(value, ','.join(set(other_types))))
    return device_lst