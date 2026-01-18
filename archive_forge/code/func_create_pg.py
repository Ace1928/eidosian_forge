from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.snapshots import (
def create_pg(module, fusion):
    """Create Placement Group"""
    pg_api_instance = purefusion.PlacementGroupsApi(fusion)
    if not module.check_mode:
        if not module.params['display_name']:
            display_name = module.params['name']
        else:
            display_name = module.params['display_name']
        group = purefusion.PlacementGroupPost(availability_zone=module.params['availability_zone'], name=module.params['name'], display_name=display_name, region=module.params['region'], storage_service=module.params['storage_service'])
        op = pg_api_instance.create_placement_group(group, tenant_name=module.params['tenant'], tenant_space_name=module.params['tenant_space'])
        res_op = await_operation(fusion, op)
        id = res_op.result.resource.id
    return (True, id)