from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def handle_login(module):
    """ Test user credentials by logging in """
    system = get_system(module)
    user_name = module.params['user_name']
    user_password = module.params['user_password']
    path = 'users/login'
    data = {'username': user_name, 'password': user_password}
    try:
        login = system.api.post(path=path, data=data)
    except APICommandFailed:
        msg = f'User {user_name} failed to login'
        module.fail_json(msg=msg)
    if login.status_code == 200:
        msg = f'User {user_name} successfully logged in'
        module.exit_json(changed=False, msg=msg)
    else:
        msg = f'User {user_name} failed to login with status code: {login.status_code}'
        module.fail_json(msg=msg)