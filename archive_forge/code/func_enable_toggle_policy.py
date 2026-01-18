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
def enable_toggle_policy(module, rest_obj, policies):
    enabler = module.params.get('enable')
    id_list = [x.get('Id') for x in policies if x.get('Enabled') is not enabler]
    if not id_list:
        module.exit_json(msg=NO_CHANGES_MSG)
    if module.check_mode:
        module.exit_json(msg=CHANGES_MSG, changed=True)
    uri = ENABLE_URI if enabler else DISABLE_URI
    rest_obj.invoke_request('POST', uri, data={'AlertPolicyIds': id_list})
    module.exit_json(changed=True, msg=SUCCESS_MSG.format('enable' if enabler else 'disable'))