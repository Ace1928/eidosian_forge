from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_symlink_mapping_rest(self, modify):
    """
        Updates a specific UNIX symbolink mapping for a SVM
        """
    api = 'protocols/cifs/unix-symlink-mapping/%s/%s' % (self.svm_uuid, self.encode_path(self.parameters['unix_path']))
    body = {'target': {}}
    for key, option in [('share', 'share_name'), ('path', 'cifs_path'), ('server', 'cifs_server'), ('locality', 'locality'), ('home_directory', 'home_directory')]:
        if modify.get(option) is not None:
            body['target'][key] = modify[option]
    dummy, error = rest_generic.patch_async(self.rest_api, api, uuid_or_name=None, body=body)
    if error:
        self.module.fail_json(msg='Error while modifying cifs unix symlink mapping: %s.' % to_native(error), exception=traceback.format_exc())