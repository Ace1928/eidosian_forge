from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def build_power_state_payload(device_id, device_type, valid_option):
    """Build the payload for requested device."""
    payload = {'Id': 0, 'JobName': 'DeviceAction_Task_PowerState', 'JobDescription': 'DeviceAction_Task', 'Schedule': 'startnow', 'State': 'Enabled', 'JobType': {'Id': 3, 'Name': 'DeviceAction_Task'}, 'Params': [{'Key': 'operationName', 'Value': 'POWER_CONTROL'}, {'Key': 'powerState', 'Value': str(valid_option)}], 'Targets': [{'Id': int(device_id), 'Data': '', 'TargetType': {'Id': device_type, 'Name': 'DEVICE'}}]}
    return payload