from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def directory_user(module, rest_obj):
    user = get_directory_user(module, rest_obj)
    new_role_id = get_role(module, rest_obj)
    dir_id = get_directory(module, rest_obj)
    domain_resp, local_msg, msg = (None, '', '')
    if user is None:
        obj_gui_id, common_name = search_directory(module, rest_obj, dir_id)
        if module.check_mode:
            module.exit_json(msg=CHANGES_FOUND, changed=True)
        payload = [{'UserTypeId': 2, 'DirectoryServiceId': dir_id, 'Description': None, 'Name': common_name, 'Password': '', 'UserName': common_name, 'RoleId': new_role_id, 'Locked': False, 'IsBuiltin': False, 'Enabled': True, 'ObjectGuid': obj_gui_id}]
        domain_resp = rest_obj.invoke_request('POST', IMPORT_ACC_PRV, data=payload)
        local_msg, msg = ('import', 'imported')
    elif int(user['RoleId']) == new_role_id:
        user = rest_obj.strip_substr_dict(user)
        module.exit_json(msg=NO_CHANGES_MSG, domain_user_status=user)
    else:
        payload = {'Id': str(user['Id']), 'UserTypeId': 2, 'DirectoryServiceId': dir_id, 'UserName': user['UserName'], 'RoleId': str(new_role_id), 'Enabled': user['Enabled']}
        update_uri = "{0}('{1}')".format(ACCOUNT_URI, user['Id'])
        if module.check_mode:
            module.exit_json(msg=CHANGES_FOUND, changed=True, domain_user_status=payload)
        domain_resp = rest_obj.invoke_request('PUT', update_uri, data=payload)
        local_msg, msg = ('update', 'updated')
    if domain_resp is None:
        module.fail_json(msg='Unable to {0} the domain user group.'.format(local_msg))
    return (domain_resp.json_data, msg)