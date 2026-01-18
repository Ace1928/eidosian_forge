from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def check_local_user_manager_state(self):
    user_account = self.find_user_account()
    if not user_account:
        return 'absent'
    else:
        return 'present'