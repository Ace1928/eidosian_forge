from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def delete_users_repository(module):
    """Delete repo."""
    system = get_system(module)
    name = module.params['name']
    changed = False
    if not module.check_mode:
        repo = get_users_repository(module, disable_fail=True)
        if repo and len(repo) == 1:
            path = f'config/ldap/{repo[0]['id']}'
            try:
                system.api.delete(path=path)
                changed = True
            except APICommandFailed as err:
                if err.status_code != 404:
                    msg = f'Deletion of users repository {name} failed: {str(err)}'
                    module.fail_json(msg=msg)
    return changed