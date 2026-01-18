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
def get_items_to_remove(filter_param, return_param_tuple, return_dict, all_items, mset):
    collector = set()
    for dev in all_items:
        k = dev.get(filter_param)
        if k in mset:
            for v in return_param_tuple:
                return_dict[v].append(dev.get(v))
            collector.add(k)
    return collector