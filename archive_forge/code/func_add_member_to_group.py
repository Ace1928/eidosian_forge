from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
def add_member_to_group(module, rest_obj, group_id, device_id, key):
    group_device = rest_obj.get_all_report_details('{0}({1})/Devices'.format(GROUP_URI, group_id))
    device_exists, device_not_exists, added_ips = ([], [], [])
    if key != 'IPAddresses':
        for each in device_id:
            each_device = list(filter(lambda d: d['Id'] in [each], group_device['report_list']))
            if each_device:
                tag_or_id = each_device[0][key] if key == 'DeviceServiceTag' else each
                device_exists.append(str(tag_or_id))
            else:
                device_not_exists.append(each)
    else:
        already_existing_id = []
        for device in group_device['report_list']:
            if device['Id'] in device_id:
                device_exists.append(device_id[device['Id']])
                already_existing_id.append(device['Id'])
        device_not_exists = list(set(device_id.keys()) - set(already_existing_id))
        added_ips = [ip for d_id, ip in device_id.items() if d_id in device_not_exists]
    if module.check_mode and device_not_exists:
        module.exit_json(msg='Changes found to be applied.', changed=True, group_id=group_id)
    elif module.check_mode and (not device_not_exists):
        module.exit_json(msg='No changes found to be applied.', group_id=group_id)
    if device_exists and (not device_not_exists):
        module.exit_json(msg='No changes found to be applied.', group_id=group_id)
    payload = {'GroupId': group_id, 'MemberDeviceIds': device_not_exists}
    response = rest_obj.invoke_request('POST', ADD_MEMBER_URI, data=payload)
    return (response, added_ips)