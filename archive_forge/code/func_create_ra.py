from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
def create_ra(module, fusion):
    """Create Role Assignment"""
    ra_api_instance = purefusion.RoleAssignmentsApi(fusion)
    changed = True
    id = None
    if not module.check_mode:
        principal = get_principal(module, fusion)
        scope = get_scope(module.params)
        assignment = purefusion.RoleAssignmentPost(scope=scope, principal=principal)
        op = ra_api_instance.create_role_assignment(assignment, role_name=module.params['role'])
        res_op = await_operation(fusion, op)
        id = res_op.result.resource.id
    module.exit_json(changed=changed, id=id)