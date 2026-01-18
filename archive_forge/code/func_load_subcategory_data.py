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
def load_subcategory_data(module, inp_sub_cat_list, sub_cat_dict, key_id, payload_cat, payload_subcat, inp_category):
    if inp_sub_cat_list:
        for sub_cat in inp_sub_cat_list:
            if sub_cat in sub_cat_dict:
                payload_cat.append(key_id)
                payload_subcat.append(sub_cat_dict.get(sub_cat))
            else:
                module.exit_json(failed=True, msg=SUBCAT_IN_CATEGORY.format(sub_cat, inp_category.get('category_name')))
    else:
        payload_cat.append(key_id)
        payload_subcat.append(0)