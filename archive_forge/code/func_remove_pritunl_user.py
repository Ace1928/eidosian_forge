from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.net_tools.pritunl.api import (
def remove_pritunl_user(module):
    result = {}
    org_name = module.params.get('organization')
    user_name = module.params.get('user_name')
    org_obj_list = []
    org_obj_list = list_pritunl_organizations(**dict_merge(get_pritunl_settings(module), {'filters': {'name': org_name}}))
    if len(org_obj_list) == 0:
        module.fail_json(msg="Can not remove user '%s' from a non existing organization '%s'" % (user_name, org_name))
    org_id = org_obj_list[0]['id']
    users = list_pritunl_users(**dict_merge(get_pritunl_settings(module), {'organization_id': org_id, 'filters': {'name': user_name}}))
    if len(users) == 0:
        result['changed'] = False
        result['response'] = {}
    else:
        response = delete_pritunl_user(**dict_merge(get_pritunl_settings(module), {'organization_id': org_id, 'user_id': users[0]['id']}))
        result['changed'] = True
        result['response'] = response
    module.exit_json(**result)