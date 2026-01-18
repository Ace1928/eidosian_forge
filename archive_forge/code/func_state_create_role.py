from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def state_create_role(self):
    """Create local role"""
    role_id = None
    results = dict()
    results['role_name'] = self.role_name
    results['privileges'] = self.priv_ids
    results['local_role_name'] = self.role_name
    results['new_privileges'] = self.priv_ids
    if self.module.check_mode:
        results['msg'] = 'Role would be created'
    else:
        try:
            role_id = self.content.authorizationManager.AddAuthorizationRole(name=self.role_name, privIds=self.priv_ids)
            results['role_id'] = role_id
            results['msg'] = 'Role created'
        except vim.fault.AlreadyExists as already_exists:
            self.module.fail_json(msg="Failed to create role '%s' as the user specified role name already exists." % self.role_name, details=already_exists.msg)
        except vim.fault.InvalidName as invalid_name:
            self.module.fail_json(msg='Failed to create a role %s as the user specified role name is empty' % self.role_name, details=invalid_name.msg)
        except vmodl.fault.InvalidArgument as invalid_argument:
            self.module.fail_json(msg='Failed to create a role %s as the user specified privileges are unknown' % self.role_name, etails=invalid_argument.msg)
    self.module.exit_json(changed=True, result=results)