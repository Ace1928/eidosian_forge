from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def handle_absent(module):
    """Make users repository absent"""
    name = module.params['name']
    msg = f'Users repository {name} unchanged'
    changed = False
    if not module.check_mode:
        changed = delete_users_repository(module)
        if changed:
            msg = f'Users repository {name} removed'
        else:
            msg = f'Users repository {name} did not exist so removal was unnecessary'
    module.exit_json(changed=changed, msg=msg)