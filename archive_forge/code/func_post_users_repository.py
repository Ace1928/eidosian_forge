from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def post_users_repository(module):
    """
    Create or update users LDAP or AD repo. The changed variable is found elsewhere.
    Variable 'changed' not returned by design
    """
    system = get_system(module)
    name = module.params['name']
    data = create_post_data(module)
    path = 'config/ldap'
    try:
        system.api.post(path=path, data=data)
    except APICommandFailed as err:
        if err.error_code == 'LDAP_NAME_CONFLICT':
            msg = f'Users repository {name} conflicts.'
            module.fail_json(msg=msg)
        elif err.error_code == 'LDAP_BAD_CREDENTIALS':
            msg = f'Cannot create users repository {name} due to incorrect LDAP credentials: {err}'
            module.fail_json(msg=msg)
        else:
            msg = f'Cannot create users repository {name}: {err}'
            module.fail_json(msg=msg)