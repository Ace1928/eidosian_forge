from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def state_remove_user(self):
    try:
        self.content.accountManager.RemoveUser(self.local_user_name)
        self.module.exit_json(changed=True)
    except vmodl.RuntimeFault as runtime_fault:
        self.module.fail_json(msg=runtime_fault.msg)
    except vmodl.MethodFault as method_fault:
        self.module.fail_json(msg=method_fault.msg)