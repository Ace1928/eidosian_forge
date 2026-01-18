from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import apply_diff_key, job_tracking
def get_template_by_name(template_name, module, rest_obj):
    template = {}
    template_path = TEMPLATES_URI
    query_param = {'$filter': "Name eq '{0}'".format(template_name)}
    template_req = rest_obj.invoke_request('GET', template_path, query_param=query_param)
    for each in template_req.json_data.get('value'):
        if each['Name'] == template_name:
            template = each
            break
    return template