from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def create_ldap_user_group(module):
    """ Create ldap user group """
    ldap_group_name = module.params['user_ldap_group_name']
    ldap_name = module.params['user_ldap_group_ldap']
    ldap_id = find_ldap_id(module)
    ldap_pools = module.params['user_ldap_group_pools']
    if not ldap_id:
        msg = f'Cannot create LDAP group {ldap_group_name}. Cannot find ID for LDAP name {ldap_name}'
        module.fail_json(msg=msg)
    path = 'users'
    system = get_system(module)
    data = {'name': ldap_group_name, 'dn': module.params['user_ldap_group_dn'], 'ldap_id': ldap_id, 'role': module.params['user_ldap_group_role'], 'type': 'Ldap'}
    try:
        system.api.post(path=path, data=data)
    except APICommandFailed as err:
        if err.status_code in [409]:
            msg = f'Cannot create user_ldap_group_name {ldap_group_name}: {err.message}'
            module.fail_json(msg)
    changed = True
    user = get_user(module, system, ldap_group_name)
    for pool_name in ldap_pools:
        pool = system.pools.get(name=pool_name)
        add_user_to_pool_owners(user, pool)
    return changed