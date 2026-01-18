from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.common.dict_transformations import recursive_diff
def check_existing_baseline(module, rest_obj):
    baseline_id = module.params.get('baseline_id')
    srch_key = 'Name'
    srch_val = module.params.get('baseline_name')
    if baseline_id:
        srch_key = 'Id'
        srch_val = module.params.get('baseline_id')
    baseline_cfgs = []
    resp = rest_obj.get_all_items_with_pagination(BASELINE_URI)
    baselines = resp.get('value')
    for d in baselines:
        if d[srch_key] == srch_val:
            baseline_cfgs.append(d)
            if baseline_id:
                break
    return baseline_cfgs