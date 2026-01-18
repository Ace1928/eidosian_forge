from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.common.dict_transformations import recursive_diff
def get_target_details(module, rest_obj):
    id = module.params.get('device_id')
    query_param = {'$filter': 'Id eq {0}'.format(id)}
    srch = 'Id'
    if not id:
        id = module.params.get('device_service_tag')
        query_param = {'$filter': "Identifier eq '{0}'".format(id)}
        srch = 'Identifier'
    resp = rest_obj.invoke_request('GET', DEVICE_VIEW, query_param=query_param)
    if resp.success and resp.json_data.get('value'):
        tlist = resp.json_data.get('value', [])
        for xtype in tlist:
            if xtype.get(srch) == id:
                return xtype
    return "Target with {0} '{1}' not found.".format(srch, id)