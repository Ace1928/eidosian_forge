from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.common.dict_transformations import recursive_diff
def get_catrepo_ids(module, cat_name, rest_obj):
    if cat_name is not None:
        resp_data = rest_obj.get_all_items_with_pagination(CATALOG_URI)
        values = resp_data['value']
        if values:
            for catalog in values:
                repo = catalog.get('Repository')
                if repo.get('Name') == cat_name:
                    if catalog.get('Status') != 'Completed':
                        module.fail_json(msg=CATALOG_STATUS_MESSAGE.format(status=catalog.get('Status')))
                    return (catalog.get('Id'), repo.get('Id'))
    return (None, None)