from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import apply_diff_key, job_tracking
def get_view_id(rest_obj, viewstr):
    resp = rest_obj.invoke_request('GET', 'TemplateService/TemplateViewTypes')
    if resp.success and resp.json_data.get('value'):
        tlist = resp.json_data.get('value', [])
        for xtype in tlist:
            if xtype.get('Description', '') == viewstr:
                return xtype.get('Id')
    viewmap = {'Deployment': 2, 'Compliance': 1, 'Inventory': 3, 'Sample': 4, 'None': 0}
    return viewmap.get(viewstr)