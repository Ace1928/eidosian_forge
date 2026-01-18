from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def is_modify_acl_required(self, desired_acl, current_acl):
    current_acl_copy = current_acl.copy()
    current_acl_copy.pop('user')
    modify = self.na_helper.get_modified_attributes(current_acl_copy, desired_acl)
    return bool(modify)