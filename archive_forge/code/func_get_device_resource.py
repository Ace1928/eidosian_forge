from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def get_device_resource(module, rest_obj):
    """Getting the device id filtered from the device inventory."""
    power_state = module.params['power_state']
    device_id = module.params['device_id']
    service_tag = module.params['device_service_tag']
    resp_data = rest_obj.get_all_report_details('DeviceService/Devices')
    if resp_data['report_list'] and service_tag is not None:
        device_resp = dict([(device.get('DeviceServiceTag'), str(device.get('Id'))) for device in resp_data['report_list']])
        if service_tag in device_resp:
            device_id = device_resp[service_tag]
        else:
            module.fail_json(msg="Unable to complete the operation because the entered target device service tag '{0}' is invalid.".format(service_tag))
    current_state, device_type = get_device_state(module, resp_data, device_id)
    valid_option, valid_operation = (VALID_OPERATION[power_state], False)
    if power_state in NOT_APPLICABLE_OPTIONS and current_state != POWER_STATE_MAP['on']:
        valid_operation = True
    elif valid_option == current_state or (power_state == 'on' and current_state in (POWER_STATE_MAP['on'], POWER_STATE_MAP['poweringon'])) or (power_state in ('off', 'shutdown') and current_state in (POWER_STATE_MAP['off'], POWER_STATE_MAP['poweringoff'])):
        valid_operation = True
    if module.check_mode and valid_operation:
        module.exit_json(msg='No changes found to commit.')
    elif module.check_mode and (not valid_operation):
        module.exit_json(msg='Changes found to commit.', changed=True)
    payload = build_power_state_payload(device_id, device_type, valid_option)
    return payload