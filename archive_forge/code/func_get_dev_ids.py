from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict, job_tracking
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import CHANGES_MSG, NO_CHANGES_MSG
def get_dev_ids(module, rest_obj, types):
    invalids = set()
    sts = module.params.get('device_ids')
    param = '{0} eq {1}'
    srch = 'Id'
    if not sts:
        sts = module.params.get('device_service_tags')
        param = "{0} eq '{1}'"
        srch = 'Identifier'
    devs = []
    for st in sts:
        resp = rest_obj.invoke_request('GET', DEVICE_URI, query_param={'$filter': param.format(srch, st)})
        val = resp.json_data.get('value')
        if not val:
            invalids.add(st)
        for v in val:
            if v[srch] == st:
                if v['Type'] in types:
                    devs.extend(val)
                else:
                    invalids.add(st)
                break
        else:
            invalids.add(st)
    valids = [dv.get('Id') for dv in devs]
    return (valids, invalids)