from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils import getters
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def create_ss(module, fusion):
    """Create Storage Service"""
    ss_api_instance = purefusion.StorageServicesApi(fusion)
    changed = True
    id = None
    if not module.check_mode:
        if not module.params['display_name']:
            display_name = module.params['name']
        else:
            display_name = module.params['display_name']
        s_service = purefusion.StorageServicePost(name=module.params['name'], display_name=display_name, hardware_types=module.params['hardware_types'])
        op = ss_api_instance.create_storage_service(s_service)
        res_op = await_operation(fusion, op)
        id = res_op.result.resource.id
    module.exit_json(changed=changed, id=id)