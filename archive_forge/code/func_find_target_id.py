from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def find_target_id(module, system):
    """ Find the ID of the target by name """
    target_name = module.params['name']
    try:
        path = f'notifications/targets?name={target_name}&fields=id'
        api_result = system.api.get(path=path)
    except APICommandFailed as err:
        msg = f'Cannot find ID for notification target {target_name}: {err}'
        module.fail_json(msg=msg)
    if len(api_result.get_json()['result']) > 0:
        result = api_result.get_json()['result'][0]
        target_id = result['id']
    else:
        target_id = None
    return target_id