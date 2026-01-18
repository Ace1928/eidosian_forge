from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def search_directory(module, rest_obj, dir_id):
    group_name, obj_gui_id, common_name = (module.params['group_name'], None, None)
    payload = {'DirectoryServerId': dir_id, 'Type': module.params['directory_type'], 'UserName': module.params['domain_username'], 'Password': module.params['domain_password'], 'CommonName': group_name}
    try:
        resp = rest_obj.invoke_request('POST', SEARCH_GROUPS, data=payload)
        for key in resp.json_data:
            if key['CommonName'].lower() == group_name.lower():
                obj_gui_id = key['ObjectGuid']
                common_name = key['CommonName']
                break
        else:
            module.fail_json(msg="Unable to complete the operation because the entered group name '{0}' does not exist.".format(group_name))
    except HTTPError as err:
        error = json.load(err)
        if error['error']['@Message.ExtendedInfo'][0]['MessageId'] in ['CGEN1004', 'CSEC5022']:
            module.fail_json(msg='Unable to complete the operation because the entered domain username or domain password are invalid.')
    return (obj_gui_id, common_name)