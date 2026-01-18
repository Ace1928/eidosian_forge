from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def check_local_role_manager_state(self):
    """Check local roles"""
    auth_role = self.find_authorization_role()
    if auth_role:
        self.current_role = auth_role
        return 'present'
    return 'absent'