from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def handle_reset_password(module):
    """ Reset user password """
    system = get_system(module)
    user = get_user(module, system)
    user_name = module.params['user_name']
    if not user:
        msg = f'Cannot change password. User {user_name} not found'
        module.fail_json(msg=msg)
    else:
        reset_user_password(module, user)
        msg = f'User {user_name} password changed'
        module.exit_json(changed=True, msg=msg)