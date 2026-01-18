from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def login_security_setting(module, rest_obj):
    security_set, attr_dict = get_security_payload(rest_obj)
    new_attr_dict = compare_merge(module, attr_dict)
    comps = security_set.get('SystemConfiguration', {}).get('Components', [{'Attributes': []}])
    comps[0]['Attributes'] = [{'Name': k, 'Value': v} for k, v in new_attr_dict.items()]
    resp = rest_obj.invoke_request('POST', SET_SETTINGS, data=security_set)
    job_id = resp.json_data.get('JobId')
    exit_settings(module, rest_obj, job_id)