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
def get_category_payloadlist(module, inp_catalog_list, cdict_ref):
    payload_cat_list = []
    for inp_catalog in inp_catalog_list:
        new_dict = {}
        catalog_name = inp_catalog.get('catalog_name')
        if catalog_name in cdict_ref:
            new_dict['CatalogName'] = catalog_name
            payload_cat = []
            category_det = cdict_ref.get(catalog_name)
            payload_subcat = []
            category_list = inp_catalog.get('catalog_category')
            load_category_data(module, catalog_name, category_list, category_det, payload_cat, payload_subcat)
            new_dict['Categories'] = payload_cat
            new_dict['SubCategories'] = payload_subcat
        else:
            module.exit_json(failed=True, msg=CATALOG_DIS_EXIST.format(catalog_name))
        payload_cat_list.append(new_dict)
    return payload_cat_list