from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def remove_ntfs_sd(self):
    """
        Deletes a NTFS security descriptor
        """
    ntfs_sd_obj = netapp_utils.zapi.NaElement('file-directory-security-ntfs-delete')
    ntfs_sd_obj.add_new_child('ntfs-sd', self.parameters['name'])
    try:
        self.server.invoke_successfully(ntfs_sd_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error deleting NTFS security descriptor %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())