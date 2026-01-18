from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_symlink_mapping_rest(self):
    """
        Retrieves a specific UNIX symbolink mapping for a SVM
        """
    api = 'protocols/cifs/unix-symlink-mapping'
    query = {'svm.name': self.parameters.get('vserver'), 'unix_path': self.parameters['unix_path'], 'fields': 'svm.uuid,unix_path,target.share,target.path,'}
    if self.parameters.get('cifs_server') is not None:
        query['fields'] += 'target.server,'
    if self.parameters.get('locality') is not None:
        query['fields'] += 'target.locality,'
    if self.parameters.get('home_directory') is not None:
        query['fields'] += 'target.home_directory,'
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    if error:
        self.module.fail_json(msg='Error while fetching cifs unix symlink mapping: %s' % to_native(error), exception=traceback.format_exc())
    if record:
        self.svm_uuid = self.na_helper.safe_get(record, ['svm', 'uuid'])
        return self.format_record(record)
    return None