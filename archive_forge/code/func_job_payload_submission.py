from __future__ import (absolute_import, division, print_function)
import copy
import json
import socket
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def job_payload_submission(rest_obj, payload, slot_payload, settings_type, device_id, resp_data):
    job_params = []
    job_params.append({'Key': 'protocolTypeV4', 'Value': payload['ProtocolTypeV4']})
    job_params.append({'Key': 'protocolTypeV6', 'Value': payload['ProtocolTypeV6']})
    s_type = 'SERVER_QUICK_DEPLOY' if settings_type == 'ServerQuickDeploy' else 'IOM_QUICK_DEPLOY'
    job_params.append({'Key': 'operationName', 'Value': '{0}'.format(s_type)})
    job_params.append({'Key': 'deviceId', 'Value': '{0}'.format(device_id)})
    if payload.get('rootCredential') is not None:
        job_params.append({'Key': 'rootCredential', 'Value': payload['rootCredential']})
    if payload.get('NetworkTypeV4') is not None:
        job_params.append({'Key': 'networkTypeV4', 'Value': payload['NetworkTypeV4']})
    if payload.get('IpV4SubnetMask') is not None:
        job_params.append({'Key': 'subnetMaskV4', 'Value': payload['IpV4SubnetMask']})
    if payload.get('IpV4Gateway') is not None:
        job_params.append({'Key': 'gatewayV4', 'Value': payload['IpV4Gateway']})
    if payload.get('NetworkTypeV6') is not None:
        job_params.append({'Key': 'networkTypeV6', 'Value': payload['NetworkTypeV6']})
    if payload.get('PrefixLength') is not None:
        job_params.append({'Key': 'prefixLength', 'Value': payload['PrefixLength']})
    if payload.get('IpV6Gateway') is not None:
        job_params.append({'Key': 'gatewayV6', 'Value': payload['IpV6Gateway']})
    updated_slot = []
    if slot_payload:
        for each in slot_payload:
            updated_slot.append(each.get('SlotId'))
            job_params.append({'Key': 'slotId={0}'.format(each.get('SlotId')), 'Value': 'SlotSelected=true;IPV4Address={0};IPV6Address={1};VlanId={2}'.format(each.get('SlotIPV4Address'), each.get('SlotIPV6Address'), each.get('VlanId'))})
    slots = resp_data['Slots']
    if updated_slot is not None:
        slots = list(filter(lambda d: d['SlotId'] not in updated_slot, slots))
    for each in slots:
        key = 'slot_id={0}'.format(each['SlotId'])
        value = 'SlotSelected={0};'.format(each['SlotSelected'])
        if each.get('SlotIPV4Address') is not None:
            value = value + 'IPV4Address={0};'.format(each['SlotIPV4Address'])
        if each.get('SlotIPV6Address') is not None:
            value = value + 'IPV6Address={0};'.format(each['SlotIPV6Address'])
        if each.get('VlanId') is not None:
            value = value + 'VlanId={0}'.format(each['VlanId'])
        job_params.append({'Key': key, 'Value': value})
    job_sub_resp = rest_obj.job_submission('Quick Deploy', QUICK_DEPLOY_JOB_DESC, [], job_params, {'Id': 42, 'Name': 'QuickDeploy_Task'})
    return job_sub_resp.json_data.get('Id')