from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import raise_from
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def remove_role_perm(self):
    role_permission = self._get_rule()
    if role_permission:
        self.result['changed'] = True
        args = {'id': role_permission['id']}
        if not self.module.check_mode:
            self.query_api('deleteRolePermission', **args)
    return role_permission