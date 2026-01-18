from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def get_ntfs_sd(self):
    ntfs_sd_entry, result = (None, None)
    ntfs_sd_get_iter = netapp_utils.zapi.NaElement('file-directory-security-ntfs-get-iter')
    ntfs_sd_info = netapp_utils.zapi.NaElement('file-directory-security-ntfs')
    ntfs_sd_info.add_new_child('vserver', self.parameters['vserver'])
    ntfs_sd_info.add_new_child('ntfs-sd', self.parameters['name'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(ntfs_sd_info)
    ntfs_sd_get_iter.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(ntfs_sd_get_iter, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching NTFS security descriptor %s : %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        attributes_list = result.get_child_by_name('attributes-list')
        ntfs_sd = attributes_list.get_child_by_name('file-directory-security-ntfs')
        ntfs_sd_entry = {'vserver': ntfs_sd.get_child_content('vserver'), 'name': ntfs_sd.get_child_content('ntfs-sd'), 'owner': ntfs_sd.get_child_content('owner'), 'group': ntfs_sd.get_child_content('group'), 'control_flags_raw': ntfs_sd.get_child_content('control-flags-raw')}
        if ntfs_sd_entry.get('control_flags_raw'):
            ntfs_sd_entry['control_flags_raw'] = int(ntfs_sd_entry['control_flags_raw'])
        return ntfs_sd_entry
    return None