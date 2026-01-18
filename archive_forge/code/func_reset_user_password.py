from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def reset_user_password(module, user):
    """ Reset user's password """
    if user is None:
        module.fail_json(msg=f'Cannot change user {module.params['user_name']} password. User not found.')
    user.update_password(module.params['user_password'])