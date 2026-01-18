from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_nfs_service_rest(self, modify):
    if self.svm_uuid is None:
        self.module.fail_json(msg='Error modifying nfs service for SVM %s: svm.uuid is None' % self.parameters['vserver'])
    api = 'protocols/nfs/services'
    body = {}
    body.update(self.create_modify_body(body, modify))
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.svm_uuid, body, job_timeout=120)
    if error:
        self.module.fail_json(msg='Error modifying nfs service for SVM %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())