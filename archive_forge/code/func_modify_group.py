from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def modify_group(rest_obj, module, valid_group_dict, parent, static_root):
    if not is_valid_static_group(valid_group_dict):
        module.fail_json(msg=INVALID_GROUPS_MODIFY)
    grp = valid_group_dict
    diff = 0
    payload = dict([(k, grp.get(k)) for k in ['Name', 'Description', 'MembershipTypeId', 'ParentId', 'Id']])
    new_name = module.params.get('new_name')
    if new_name:
        if new_name != payload['Name']:
            dup_grp = get_ome_group_by_name(rest_obj, new_name)
            if dup_grp:
                module.fail_json(msg=GROUP_NAME_EXISTS.format(gname=new_name))
            payload['Name'] = new_name
            diff += 1
    desc = module.params.get('description')
    if desc:
        if desc != payload['Description']:
            payload['Description'] = desc
            diff += 1
    parent_id = get_parent_id(rest_obj, module, parent, static_root)
    if parent_id == payload['Id']:
        module.fail_json(msg=GROUP_PARENT_SAME)
    if parent_id != payload['ParentId']:
        payload['ParentId'] = parent_id
        diff += 1
    if diff == 0:
        gs = rest_obj.strip_substr_dict(grp)
        module.exit_json(msg=NO_CHANGES_MSG, group_status=gs)
    if module.check_mode:
        module.exit_json(changed=True, msg=CHANGES_FOUND)
    exit_group_operation(module, rest_obj, payload, 'Update')