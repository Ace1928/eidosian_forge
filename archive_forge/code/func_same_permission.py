from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_obj
def same_permission(self, perm_one, perm_two):
    return perm_one.principal.lower() == perm_two.principal.lower() and perm_one.roleId == perm_two.roleId