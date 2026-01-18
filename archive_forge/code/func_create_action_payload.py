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
def create_action_payload(inp_k, inp_val, ref_actions, module):
    if ref_actions.get(inp_k).get('Disabled'):
        module.exit_json(failed=True, msg=DISABLED_ACTION.format(inp_k))
    pld = {'TemplateId': ref_actions.get(inp_k).get('Id'), 'Name': inp_k, 'ParameterDetails': {}}
    diff = set(inp_val.keys()) - set(ref_actions.get(inp_k).get('Parameters').keys())
    if diff:
        module.exit_json(failed=True, msg=ACTION_INVALID_PARAM.format(inp_k, SEPARATOR.join(diff), SEPARATOR.join(ref_actions.get(inp_k).get('Parameters').keys())))
    for sub_k, sub_val in inp_val.items():
        valid_values = ref_actions.get(inp_k).get('Type').get(sub_k)
        if valid_values:
            if str(sub_val).lower() not in valid_values:
                module.exit_json(failed=True, msg=ACTION_INVALID_VALUE.format(inp_k, sub_val, sub_k, SEPARATOR.join(valid_values)))
            else:
                inp_val[sub_k] = str(sub_val).lower() if str(sub_val).lower() in ('true', 'false') else sub_val
    pld['ParameterDetails'] = inp_val
    return pld