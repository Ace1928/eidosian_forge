from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils import getters
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def delete_ss(module, fusion):
    """Delete Storage Service"""
    ss_api_instance = purefusion.StorageServicesApi(fusion)
    changed = True
    if not module.check_mode:
        op = ss_api_instance.delete_storage_service(storage_service_name=module.params['name'])
        await_operation(fusion, op)
    module.exit_json(changed=changed)