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
def get_category_or_message(module, rest_obj):
    cat_payload = {'Catalogs': {}, 'MessageIds': []}
    cat_msg_provided = False
    if module.params.get('category'):
        payload_cat_list = get_category_payload(module, rest_obj)
        cat_dict = dict(((x.get('CatalogName'), x) for x in payload_cat_list))
        cat_msg_provided = True
        cat_payload['Catalogs'] = cat_dict
    else:
        mlist = get_message_payload(module)
        if mlist:
            validate_ome_data(module, rest_obj, mlist, 'MessageId', ('MessageId',), MESSAGES_URI, 'messages')
            cat_msg_provided = True
            cat_payload['MessageIds'] = list(set(mlist))
            cat_payload['MessageIds'].sort()
    if not cat_msg_provided:
        cat_payload = {}
    return cat_payload