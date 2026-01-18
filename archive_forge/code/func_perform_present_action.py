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
def perform_present_action(module, rest_obj, requested_catalog_list, all_catalog):
    if requested_catalog_list:
        modify_catalog(module, rest_obj, requested_catalog_list, all_catalog)
    else:
        if module.params.get('catalog_id'):
            module.fail_json(msg=INVALID_CATALOG_ID)
        repository_type = module.params.get('repository_type')
        if repository_type and repository_type == 'DELL_ONLINE':
            validate_dell_online(all_catalog, module)
        create_catalog(module, rest_obj)