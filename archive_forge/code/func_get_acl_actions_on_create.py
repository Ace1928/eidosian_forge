from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def get_acl_actions_on_create(self):
    """
        POST does not accept access_control and propagation_mode at the ACL level, these are global values for all ACLs.
        Since the user could have a list of ACLs with mixed property we should useP OST the create the SD and 1 group of ACLs
        then loop over the remaining ACLS.
        """
    acls_groups = {}
    preferred_group = (None, None)
    special_accesses = []
    for acl in self.parameters.get('acls', []):
        access_control = acl.get('access_control', 'file_directory')
        propagation_mode = acl.get('propagation_mode', 'propagate')
        if access_control not in acls_groups:
            acls_groups[access_control] = {}
        if propagation_mode not in acls_groups[access_control]:
            acls_groups[access_control][propagation_mode] = []
        acls_groups[access_control][propagation_mode].append(acl)
        access = acl.get('access')
        if access not in ('access_allow', 'access_deny', 'audit_success', 'audit_failure'):
            if preferred_group == (None, None):
                preferred_group = (access_control, propagation_mode)
            if preferred_group != (access_control, propagation_mode):
                self.module.fail_json(msg='Error: acl %s with access %s conflicts with other ACLs using accesses: %s with different access_control or propagation_mode: %s.' % (acl, access, special_accesses, preferred_group))
            special_accesses.append(access)
    if preferred_group == (None, None):
        for acc_key, acc_value in sorted(acls_groups.items()):
            for prop_key, prop_value in sorted(acc_value.items()):
                if prop_value:
                    preferred_group = (acc_key, prop_key)
                    break
            if preferred_group != (None, None):
                break
    create_acls = []
    acl_actions = {'patch-acls': [], 'post-acls': [], 'delete-acls': []}
    for acc_key, acc_value in sorted(acls_groups.items()):
        for prop_key, prop_value in sorted(acc_value.items()):
            if (acc_key, prop_key) == preferred_group:
                create_acls = prop_value
                self.parameters['access_control'] = acc_key
                self.parameters['propagation_mode'] = prop_key
            elif prop_value:
                acl_actions['post-acls'].extend(prop_value)
    self.parameters['acls'] = create_acls
    return acl_actions