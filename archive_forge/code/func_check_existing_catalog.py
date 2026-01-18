from __future__ import (absolute_import, division, print_function)
import json
import time
import os
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import remove_key
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def check_existing_catalog(module, rest_obj, state, name=None):
    catalog_cfgs = []
    if name:
        catalog_id = None
        catalog_name = [name]
    else:
        catalog_id = module.params.get('catalog_id')
        catalog_name = module.params.get('catalog_name')
    resp = rest_obj.get_all_items_with_pagination(CATALOG_URI)
    catalogs_detail = resp.get('value')
    all_catalog = {}
    if state == 'present':
        all_catalog = dict([(each_catalog['Repository']['Name'], each_catalog['Repository']['RepositoryType']) for each_catalog in catalogs_detail])
    for each_catalog in catalogs_detail:
        if catalog_name:
            if each_catalog['Repository']['Name'] in catalog_name:
                catalog_cfgs.append(each_catalog)
                if state == 'present':
                    break
                continue
        if catalog_id:
            if each_catalog['Id'] in catalog_id:
                catalog_cfgs.append(each_catalog)
                if state == 'present':
                    break
                continue
    return (catalog_cfgs, all_catalog)