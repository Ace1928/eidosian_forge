from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import raise_from
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def create_role_perm(self):
    role_permission = None
    self.result['changed'] = True
    args = {'rule': self.module.params.get('name'), 'description': self.module.params.get('description'), 'roleid': self._get_role_id(), 'permission': self.module.params.get('permission')}
    if not self.module.check_mode:
        res = self.query_api('createRolePermission', **args)
        role_permission = res['rolepermission']
    return role_permission