from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def get_sip_info(module, rest_obj):
    invalid, valid_service_tag, device_map = ([], [], {})
    device_id, tag = (module.params.get('device_id'), module.params.get('device_service_tag'))
    key, value = ('Id', device_id) if device_id is not None else ('DeviceServiceTag', tag)
    resp_data = rest_obj.get_all_report_details(DEVICE_URI)
    if resp_data['report_list']:
        for each in value:
            each_device = list(filter(lambda d: d[key] in [each], resp_data['report_list']))
            if each_device and key == 'DeviceServiceTag':
                valid_service_tag.append(each)
            elif each_device and key == 'Id':
                valid_service_tag.append(each_device[0]['DeviceServiceTag'])
                device_map[each_device[0]['DeviceServiceTag']] = each
            if not each_device:
                invalid.append(each)
    if invalid:
        err_value = 'id' if key == 'Id' else 'service tag'
        module.fail_json(msg=INVALID_DEVICE.format(err_value, ','.join(map(str, set(invalid)))))
    invalid_fabric_tag, sip_info = ([], [])
    for pro_id in valid_service_tag:
        profile_dict = {}
        try:
            profile_resp = rest_obj.invoke_request('GET', "{0}('{1}')".format(PROFILE_URI, pro_id))
        except HTTPError as err:
            err_message = json.load(err)
            if err_message.get('error', {}).get('@Message.ExtendedInfo')[0]['MessageId'] == 'CDEV5008':
                if key == 'Id':
                    invalid_fabric_tag.append(device_map[pro_id])
                else:
                    invalid_fabric_tag.append(pro_id)
        else:
            profile_data = rest_obj.strip_substr_dict(profile_resp.json_data)
            profile_dict.update(profile_data)
            np_resp = rest_obj.invoke_request('GET', NETWORK_PROFILE_URI.format(pro_id))
            sip_strip = []
            for each in np_resp.json_data['value']:
                np_strip_data = rest_obj.strip_substr_dict(each)
                np_strip_data['Networks'] = [rest_obj.strip_substr_dict(each) for each in np_strip_data['Networks']]
                sip_strip.append(np_strip_data)
            profile_dict['ServerInterfaceProfile'] = sip_strip
            sip_info.append(profile_dict)
    if invalid_fabric_tag:
        module.fail_json(msg=PROFILE_ERR_MSG.format(', '.join(set(map(str, invalid_fabric_tag)))))
    return sip_info