from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def delete_sso(module, sso_id):
    """ Delete a SSO. Reference its ID. """
    path = f'config/sso/idps/{sso_id}'
    name = module.params['name']
    try:
        system = get_system(module)
        sso_result = system.api.delete(path=path).get_result()
    except APICommandFailed as err:
        msg = f'Cannot delete SSO identity provider {name}: {err}'
        module.fail_json(msg=msg)
    return sso_result