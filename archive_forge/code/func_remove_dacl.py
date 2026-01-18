from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def remove_dacl(self):
    """
        Deletes a NTFS DACL from an existing NTFS security descriptor
        """
    dacl_obj = netapp_utils.zapi.NaElement('file-directory-security-ntfs-dacl-remove')
    dacl_obj.add_new_child('access-type', self.parameters['access_type'])
    dacl_obj.add_new_child('account', self.parameters['account'])
    dacl_obj.add_new_child('ntfs-sd', self.parameters['security_descriptor'])
    try:
        self.server.invoke_successfully(dacl_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error deleting %s DACL for account %s for security descriptor %s: %s' % (self.parameters['access_type'], self.parameters['account'], self.parameters['security_descriptor'], to_native(error)), exception=traceback.format_exc())