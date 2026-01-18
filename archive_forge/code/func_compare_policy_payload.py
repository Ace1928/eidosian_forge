from __future__ import (absolute_import, division, print_function)
import csv
import os
import json
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_all_data_with_pagination, strip_substr_dict
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.common.dict_transformations import recursive_diff
from datetime import datetime
def compare_policy_payload(module, rest_obj, policy):
    diff = 0
    new_payload = {}
    new_policy_data = {}
    new_payload['PolicyData'] = new_policy_data
    transform_existing_policy_data(policy)
    payload_items = []
    payload_items.append(get_target_payload(module, rest_obj))
    payload_items.append(get_category_or_message(module, rest_obj))
    payload_items.append(get_actions_payload(module, rest_obj))
    payload_items.append(get_schedule_payload(module))
    payload_items.append(get_severity_payload(module, rest_obj))
    for payload in payload_items:
        if payload:
            new_policy_data.update(payload)
            diff_tuple = recursive_diff(new_payload['PolicyData'], policy['PolicyData'])
            if diff_tuple and diff_tuple[0]:
                diff = diff + 1
                policy['PolicyData'].update(payload)
    if module.params.get('new_name'):
        new_payload['Name'] = module.params.get('new_name')
    if module.params.get('description'):
        new_payload['Description'] = module.params.get('description')
    if module.params.get('enable') is not None:
        new_payload['Enabled'] = module.params.get('enable')
    policy = strip_substr_dict(policy)
    new_payload.pop('PolicyData', None)
    diff_tuple = recursive_diff(new_payload, policy)
    if diff_tuple and diff_tuple[0]:
        diff = diff + 1
        policy.update(diff_tuple[0])
    return diff