from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def get_directory(module, rest_obj):
    user_dir_name = module.params.get('directory_name')
    user_dir_id = module.params.get('directory_id')
    key = 'name' if user_dir_name is not None else 'id'
    value = user_dir_name if user_dir_name is not None else user_dir_id
    dir_id = None
    if user_dir_name is None and user_dir_id is None:
        module.fail_json(msg='missing required arguments: directory_name or directory_id')
    URI = GET_AD_ACC if module.params.get('directory_type') == 'AD' else GET_LDAP_ACC
    directory_resp = rest_obj.invoke_request('GET', URI)
    for dire in directory_resp.json_data['value']:
        if user_dir_name is not None and dire['Name'] == user_dir_name:
            dir_id = dire['Id']
            break
        if user_dir_id is not None and dire['Id'] == user_dir_id:
            dir_id = dire['Id']
            break
    else:
        module.fail_json(msg="Unable to complete the operation because the entered directory {0} '{1}' does not exist.".format(key, value))
    return dir_id