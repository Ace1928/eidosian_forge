from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
def get_ra(module, fusion):
    """Return Role Assignment or None"""
    ra_api_instance = purefusion.RoleAssignmentsApi(fusion)
    try:
        principal = get_principal(module, fusion)
        assignments = ra_api_instance.list_role_assignments(role_name=module.params['role'], principal=principal)
        for assign in assignments:
            scope = get_scope(module.params)
            if assign.scope.self_link == scope:
                return assign
        return None
    except purefusion.rest.ApiException:
        return None