from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.net_tools.pritunl.api import (
def add_pritunl_organization(module):
    result = {}
    org_name = module.params.get('name')
    org_obj_list = list_pritunl_organizations(**dict_merge(get_pritunl_settings(module), {'filters': {'name': org_name}}))
    if len(org_obj_list) > 0:
        result['changed'] = False
        result['response'] = org_obj_list[0]
    else:
        response = post_pritunl_organization(**dict_merge(get_pritunl_settings(module), {'organization_name': org_name}))
        result['changed'] = True
        result['response'] = response
    module.exit_json(**result)