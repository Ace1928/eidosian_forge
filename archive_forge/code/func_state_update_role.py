from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def state_update_role(self):
    """Update local role"""
    changed = False
    changed_privileges = []
    results = dict()
    results['role_name'] = self.role_name
    results['role_id'] = self.current_role.roleId
    results['local_role_name'] = self.role_name
    current_privileges = self.current_role.privilege
    results['privileges'] = current_privileges
    results['new_privileges'] = current_privileges
    if self.action == 'add':
        for priv in self.params['local_privilege_ids']:
            if priv not in current_privileges:
                changed_privileges.append(priv)
                changed = True
        if changed:
            changed_privileges.extend(current_privileges)
    elif self.action == 'set':
        self.params['local_privilege_ids'].extend(['System.Anonymous', 'System.Read', 'System.View'])
        changed_privileges = self.params['local_privilege_ids']
        changes_applied = list(set(current_privileges) ^ set(changed_privileges))
        if changes_applied:
            changed = True
    elif self.action == 'remove':
        changed_privileges = list(current_privileges)
        for priv in self.params['local_privilege_ids']:
            if priv in current_privileges:
                changed = True
                changed_privileges.remove(priv)
    if changed:
        results['privileges'] = changed_privileges
        results['privileges_previous'] = current_privileges
        results['new_privileges'] = changed_privileges
        results['old_privileges'] = current_privileges
        if self.module.check_mode:
            results['msg'] = 'Role privileges would be updated'
        else:
            try:
                self.content.authorizationManager.UpdateAuthorizationRole(roleId=self.current_role.roleId, newName=self.current_role.name, privIds=changed_privileges)
                results['msg'] = 'Role privileges updated'
            except vim.fault.NotFound as not_found:
                self.module.fail_json(msg='Failed to update role. Please check privileges provided for update', details=not_found.msg)
            except vim.fault.InvalidName as invalid_name:
                self.module.fail_json(msg='Failed to update role as role name is empty', details=invalid_name.msg)
            except vim.fault.AlreadyExists as already_exists:
                self.module.fail_json(msg='Failed to update role', details=already_exists.msg)
            except vmodl.fault.InvalidArgument as invalid_argument:
                self.module.fail_json(msg='Failed to update role as user specified role is system role which can not be changed', details=invalid_argument.msg)
            except vim.fault.NoPermission as no_permission:
                self.module.fail_json(msg="Failed to update role as current session doesn't have any privilege to update specified role", details=no_permission.msg)
    else:
        results['msg'] = 'Role privileges are properly configured'
    self.module.exit_json(changed=changed, result=results)