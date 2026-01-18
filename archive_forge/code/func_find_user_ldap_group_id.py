from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def find_user_ldap_group_id(module):
    """
    Find the ID of the LDAP user group by name
    """
    ldap_id = None
    ldap_name = module.params['user_ldap_group_name']
    path = f'users?name={ldap_name}&type=eq%3ALdap'
    system = get_system(module)
    api_result = system.api.get(path=path)
    if len(api_result.get_json()['result']) > 0:
        result = api_result.get_json()['result'][0]
        ldap_id = result['id']
    return ldap_id