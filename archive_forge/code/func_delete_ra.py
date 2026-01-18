from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
def delete_ra(module, fusion):
    """Delete Role Assignment"""
    changed = True
    ra_api_instance = purefusion.RoleAssignmentsApi(fusion)
    if not module.check_mode:
        ra_name = get_ra(module, fusion).name
        op = ra_api_instance.delete_role_assignment(role_name=module.params['role'], role_assignment_name=ra_name)
        await_operation(fusion, op)
    module.exit_json(changed=changed)