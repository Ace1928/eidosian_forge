from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def modify_dacl(self):
    """
        Modifies a NTFS DACL on an existing NTFS security descriptor
        """
    dacl_obj = netapp_utils.zapi.NaElement('file-directory-security-ntfs-dacl-modify')
    dacl_obj.add_new_child('access-type', self.parameters['access_type'])
    dacl_obj.add_new_child('account', self.parameters['account'])
    dacl_obj.add_new_child('ntfs-sd', self.parameters['security_descriptor'])
    if self.parameters.get('apply_to'):
        apply_to_obj = netapp_utils.zapi.NaElement('apply-to')
        for apply_entry in self.parameters['apply_to']:
            apply_to_obj.add_new_child('inheritance-level', apply_entry)
        dacl_obj.add_child_elem(apply_to_obj)
    if self.parameters.get('advanced_access_rights'):
        access_rights_obj = netapp_utils.zapi.NaElement('advanced-rights')
        for right in self.parameters['advanced_access_rights']:
            access_rights_obj.add_new_child('advanced-access-rights', right)
        dacl_obj.add_child_elem(access_rights_obj)
    if self.parameters.get('rights'):
        dacl_obj.add_new_child('rights', self.parameters['rights'])
    try:
        self.server.invoke_successfully(dacl_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error modifying %s DACL for account %s for security descriptor %s: %s' % (self.parameters['access_type'], self.parameters['account'], self.parameters['security_descriptor'], to_native(error)), exception=traceback.format_exc())