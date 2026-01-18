from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def find_authorization_role(self):
    """Find local role"""
    desired_role = None
    for role in self.content.authorizationManager.roleList:
        if role.name == self.role_name:
            desired_role = role
    return desired_role