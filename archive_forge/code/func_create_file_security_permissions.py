from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def create_file_security_permissions(self):
    api = 'protocols/file-security/permissions/%s/%s' % (self.svm_uuid, self.url_encode(self.parameters['path']))
    body = {}
    for option in ('access_control', 'control_flags', 'group', 'owner', 'ignore_paths', 'propagation_mode'):
        self.set_option(body, option)
    body['acls'] = self.sanitize_acls_for_post(self.parameters.get('acls', []))
    dummy, error = rest_generic.post_async(self.rest_api, api, body, job_timeout=120)
    if error:
        self.module.fail_json(msg='Error creating file security permissions %s: %s' % (self.parameters['path'], to_native(error)), exception=traceback.format_exc())