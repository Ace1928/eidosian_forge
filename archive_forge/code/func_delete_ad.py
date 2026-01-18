from __future__ import (absolute_import, division, print_function)
import json
import os
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def delete_ad(module, rest_obj, ad):
    ad = rest_obj.strip_substr_dict(ad)
    if module.check_mode:
        module.exit_json(msg=CHANGES_FOUND, active_directory=ad, changed=True)
    rest_obj.invoke_request('POST', DELETE_AD, data={'AccountProviderIds': [int(ad['Id'])]})
    module.exit_json(msg=DELETE_SUCCESS, active_directory=ad, changed=True)