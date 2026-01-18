from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def populate_permission_list(self):
    results = []
    if self.principal is None:
        for permission in self.current_perms:
            results.append({'principal': permission.principal, 'role_name': self.role_list.get(permission.roleId, ''), 'role_id': permission.roleId, 'propagate': permission.propagate})
    else:
        results = self.to_json(self.current_perms)
    self.module.exit_json(changed=False, permission_info=results)