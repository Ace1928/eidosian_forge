from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.net_tools.pritunl.api import (
def add_or_update_pritunl_user(module):
    result = {}
    org_name = module.params.get('organization')
    user_name = module.params.get('user_name')
    user_params = {'name': user_name, 'email': module.params.get('user_email'), 'groups': module.params.get('user_groups'), 'disabled': module.params.get('user_disabled'), 'gravatar': module.params.get('user_gravatar'), 'mac_addresses': module.params.get('user_mac_addresses'), 'type': module.params.get('user_type')}
    org_obj_list = list_pritunl_organizations(**dict_merge(get_pritunl_settings(module), {'filters': {'name': org_name}}))
    if len(org_obj_list) == 0:
        module.fail_json(msg="Can not add user to organization '%s' which does not exist" % org_name)
    org_id = org_obj_list[0]['id']
    users = list_pritunl_users(**dict_merge(get_pritunl_settings(module), {'organization_id': org_id, 'filters': {'name': user_name}}))
    if len(users) > 0:
        user_params_changed = False
        for key in user_params.keys():
            if user_params[key] is None:
                user_params[key] = users[0][key]
            if key == 'groups' or key == 'mac_addresses':
                if set(users[0][key]) != set(user_params[key]):
                    user_params_changed = True
            elif users[0][key] != user_params[key]:
                user_params_changed = True
        if user_params_changed:
            response = post_pritunl_user(**dict_merge(get_pritunl_settings(module), {'organization_id': org_id, 'user_id': users[0]['id'], 'user_data': user_params}))
            result['changed'] = True
            result['response'] = response
        else:
            result['changed'] = False
            result['response'] = users
    else:
        response = post_pritunl_user(**dict_merge(get_pritunl_settings(module), {'organization_id': org_id, 'user_data': user_params}))
        result['changed'] = True
        result['response'] = response
    module.exit_json(**result)