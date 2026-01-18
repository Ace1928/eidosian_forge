from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def create_account_spec(self):
    account_spec = vim.host.LocalAccountManager.AccountSpecification()
    account_spec.id = self.local_user_name
    account_spec.password = self.local_user_password
    account_spec.description = self.local_user_description
    return account_spec