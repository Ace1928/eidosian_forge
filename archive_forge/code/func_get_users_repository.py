from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def get_users_repository(module, disable_fail=False):
    """
    Find and return users repository information
    Use disable_fail when we are looking for an user repository
    and it may or may not exist and neither case is an error.
    """
    system = get_system(module)
    name = module.params['name']
    path = f'config/ldap?name={name}'
    repo = system.api.get(path=path)
    if repo:
        result = repo.get_result()
        if not disable_fail and (not result):
            msg = f'Users repository {name} not found. Cannot stat.'
            module.fail_json(msg=msg)
        return result
    if not disable_fail:
        msg = f'Users repository {name} not found. Cannot stat.'
        module.fail_json(msg=msg)
    return None