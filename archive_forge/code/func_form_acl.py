from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def form_acl(acl):
    advanced_rights_keys = ['append_data', 'delete', 'delete_child', 'execute_file', 'full_control', 'read_attr', 'read_data', 'read_ea', 'read_perm', 'write_attr', 'write_data', 'write_ea', 'write_owner', 'write_perm']
    advanced_rights = {}
    apply_to = {}
    if 'advanced_rights' in acl:
        for key in advanced_rights_keys:
            advanced_rights[key] = acl['advanced_rights'].get(key, False)
    if 'apply_to' in acl:
        for key in self.apply_to_keys:
            apply_to[key] = acl['apply_to'].get(key, False)
    return {'advanced_rights': advanced_rights or None, 'apply_to': apply_to or None}