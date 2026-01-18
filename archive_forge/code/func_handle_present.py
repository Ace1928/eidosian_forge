from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def handle_present(module):
    """Make users repository present"""
    name = module.params['name']
    changed = False
    msg = ''
    if not module.check_mode:
        old_users_repo = None
        old_users_repo_result = get_users_repository(module, disable_fail=True)
        if old_users_repo_result:
            old_users_repo = old_users_repo_result[0]
            if is_existing_users_repo_equal_to_desired(module):
                msg = f'Users repository {name} already exists. No changes required.'
                module.exit_json(changed=changed, msg=msg)
            else:
                msg = f'Users repository {name} is being recreated with new settings. '
                delete_users_repository(module)
                old_users_repo = None
                changed = True
        post_users_repository(module)
        new_users_repo = get_users_repository(module)
        changed = new_users_repo != old_users_repo
        if changed:
            if old_users_repo:
                msg = f'{msg}Users repository {name} updated'
            else:
                msg = f'{msg}Users repository {name} created'
        else:
            msg = f'Users repository {name} unchanged since the value is the same as the existing users repository'
    else:
        msg = f'Users repository {name} unchanged due to check_mode'
    module.exit_json(changed=changed, msg=msg)