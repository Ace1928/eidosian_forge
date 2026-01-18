from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def handle_stat(module):
    """Return users repository stat"""
    name = module.params['name']
    repos = get_users_repository(module)
    if len(repos) != 1:
        msg = f'Users repository {name} not found in repository list {repos}. Cannot stat.'
        module.fail_json(msg=msg)
    result = repos[0]
    repository_id = result.pop('id')
    result['msg'] = f'Stats for user repository {name}'
    result['repository_id'] = repository_id
    result['test_ok'] = test_users_repository(module, repository_id=repository_id, disable_fail=True)
    result['changed'] = False
    module.exit_json(**result)