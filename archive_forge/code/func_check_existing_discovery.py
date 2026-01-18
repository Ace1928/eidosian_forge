from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def check_existing_discovery(module, rest_obj):
    discovery_cfgs = []
    discovery_id = module.params.get('discovery_id')
    srch_key = 'DiscoveryConfigGroupName'
    srch_val = module.params.get('discovery_job_name')
    if discovery_id:
        srch_key = 'DiscoveryConfigGroupId'
        srch_val = module.params.get('discovery_id')
    resp = rest_obj.invoke_request('GET', CONFIG_GROUPS_URI + '?$top=9999')
    discovs = resp.json_data.get('value')
    for d in discovs:
        if d[srch_key] == srch_val:
            discovery_cfgs.append(d)
            if discovery_id:
                break
    return discovery_cfgs