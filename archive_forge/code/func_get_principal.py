from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
def get_principal(module, fusion):
    if module.params['principal']:
        return module.params['principal']
    if module.params['user']:
        principal = user_to_principal(fusion, module.params['user'])
        if not principal:
            module.fail_json(msg='User {0} does not exist'.format(module.params['user']))
        return principal
    if module.params['api_client_key']:
        principal = apiclient_to_principal(fusion, module.params['api_client_key'])
        if not principal:
            module.fail_json(msg='API Client with key {0} does not exist'.format(module.params['api_client_key']))
        return principal