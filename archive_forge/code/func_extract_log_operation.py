from __future__ import (absolute_import, division, print_function)
import json
import re
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def extract_log_operation(module, rest_obj, device_lst=None):
    payload_params, target_params = ([], [])
    log_type = module.params['log_type']
    if log_type == 'application':
        lead_only = module.params['lead_chassis_only']
        resp_data = None
        if lead_only:
            domain_details = rest_obj.get_all_items_with_pagination(DOMAIN_URI)
            key = 'Id'
            ch_device_id = None
            for each_domain in domain_details['value']:
                if each_domain['DomainRoleTypeValue'] in ['LEAD', 'STANDALONE']:
                    ch_device_id = each_domain['DeviceId']
            if ch_device_id:
                resp = rest_obj.invoke_request('GET', DEVICE_URI, query_param={'$filter': '{0} eq {1}'.format(key, ch_device_id)})
                resp_data = resp.json_data['value']
        else:
            resp = rest_obj.invoke_request('GET', DEVICE_URI, query_param={'$filter': 'Type eq 2000'})
            resp_data = resp.json_data['value']
        if resp_data:
            for dev in resp_data:
                target_params.append({'Id': dev['Id'], 'Data': '', 'TargetType': {'Id': dev['Type'], 'Name': 'CHASSIS'}})
        else:
            module.fail_json(msg='There is no device(s) available to export application log.')
    else:
        for device in device_lst:
            target_params.append({'Id': device, 'Data': '', 'TargetType': {'Id': 1000, 'Name': 'DEVICE'}})
    payload_params.append({'Key': 'shareAddress', 'Value': module.params['share_address']})
    payload_params.append({'Key': 'shareType', 'Value': module.params['share_type']})
    payload_params.append({'Key': 'OPERATION_NAME', 'Value': 'EXTRACT_LOGS'})
    if module.params.get('share_name') is not None:
        payload_params.append({'Key': 'shareName', 'Value': module.params['share_name']})
    if module.params.get('share_user') is not None:
        payload_params.append({'Key': 'userName', 'Value': module.params['share_user']})
    if module.params.get('share_password') is not None:
        payload_params.append({'Key': 'password', 'Value': module.params['share_password']})
    if module.params.get('share_domain') is not None:
        payload_params.append({'Key': 'domainName', 'Value': module.params['share_domain']})
    if module.params.get('mask_sensitive_info') is not None and log_type == 'application':
        payload_params.append({'Key': 'maskSensitiveInfo', 'Value': str(module.params['mask_sensitive_info']).upper()})
    if module.params.get('log_selectors') is not None and (log_type == 'support_assist_collection' or log_type == 'supportassist_collection'):
        log_lst = [LOG_SELECTOR[i] for i in module.params['log_selectors']]
        log_lst.sort()
        log_selector = ','.join(map(str, log_lst))
        payload_params.append({'Key': 'logSelector', 'Value': '0,{0}'.format(log_selector)})
    response = rest_obj.job_submission('Export Log', 'Export device log', target_params, payload_params, {'Id': 18, 'Name': 'DebugLogs_Task'})
    return response