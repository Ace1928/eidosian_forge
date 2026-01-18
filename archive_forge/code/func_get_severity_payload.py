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
def get_severity_payload(module, rest_obj):
    try:
        resp = rest_obj.invoke_request('GET', SEVERITY_URI)
        severity_dict = dict(((x.get('Name').lower(), x.get('Id')) for x in resp.json_data.get('Value')))
    except Exception:
        severity_dict = {'unknown': 1, 'info': 2, 'normal': 4, 'warning': 8, 'critical': 16}
    inp_sev_list = module.params.get('severity')
    sev_payload = {}
    if inp_sev_list:
        if 'all' in inp_sev_list:
            sev_payload = {'Severities': list(severity_dict.values())}
        else:
            sev_payload = {'Severities': [severity_dict.get(x) for x in inp_sev_list]}
        sev_payload['Severities'].sort()
    return sev_payload