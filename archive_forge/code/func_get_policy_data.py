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
def get_policy_data(module, rest_obj):
    policy_data = {}
    target = get_target_payload(module, rest_obj)
    if not target:
        module.exit_json(failed=True, msg=INVALID_TARGETS)
    policy_data.update(target)
    cat_msg = get_category_or_message(module, rest_obj)
    if not cat_msg:
        module.exit_json(failed=True, msg=INVALID_CATEGORY_MESSAGE)
    policy_data.update(cat_msg)
    schedule = get_schedule_payload(module)
    if not schedule:
        module.exit_json(failed=True, msg=INVALID_SCHEDULE)
    policy_data.update(schedule)
    actions = get_actions_payload(module, rest_obj)
    if not actions:
        module.exit_json(failed=True, msg=INVALID_ACTIONS)
    policy_data.update(actions)
    sev_payload = get_severity_payload(module, rest_obj)
    if not sev_payload.get('Severities'):
        module.exit_json(failed=True, msg=INVALID_SEVERITY)
    policy_data.update(sev_payload)
    return policy_data