from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.net_tools.pritunl.api import (
def get_pritunl_user(module):
    user_name = module.params.get('user_name')
    user_type = module.params.get('user_type')
    org_name = module.params.get('organization')
    org_obj_list = []
    org_obj_list = list_pritunl_organizations(**dict_merge(get_pritunl_settings(module), {'filters': {'name': org_name}}))
    if len(org_obj_list) == 0:
        module.fail_json(msg="Can not list users from the organization '%s' which does not exist" % org_name)
    org_id = org_obj_list[0]['id']
    users = list_pritunl_users(**dict_merge(get_pritunl_settings(module), {'organization_id': org_id, 'filters': {'type': user_type} if user_name is None else {'name': user_name, 'type': user_type}}))
    result = {}
    result['changed'] = False
    result['users'] = users
    module.exit_json(**result)