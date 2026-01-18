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
def get_all_actions(rest_obj):
    resp = rest_obj.invoke_request('GET', ACTIONS_URI)
    actions = resp.json_data.get('value', [])
    cmp_actions = dict(((x.get('Name'), {'Id': x.get('Id'), 'Disabled': x.get('Disabled'), 'Parameters': dict(((y.get('Name'), y.get('Value')) for y in x.get('ParameterDetails'))), 'Type': dict(((y.get('Name'), ['true', 'false'] if y.get('Type') == 'boolean' else [z.get('Value') for z in y.get('TemplateParameterTypeDetails') if y.get('Type') != 'string']) for y in x.get('ParameterDetails')))}) for x in actions))
    return cmp_actions