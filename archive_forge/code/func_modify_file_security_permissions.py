from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def modify_file_security_permissions(self, modify):
    api = 'protocols/file-security/permissions/%s/%s' % (self.svm_uuid, self.url_encode(self.parameters['path']))
    body = {}
    for option in modify:
        self.set_option(body, option)
    dummy, error = rest_generic.patch_async(self.rest_api, api, None, body, job_timeout=120)
    if error:
        self.module.fail_json(msg='Error modifying file security permissions %s: %s' % (self.parameters['path'], to_native(error)), exception=traceback.format_exc())