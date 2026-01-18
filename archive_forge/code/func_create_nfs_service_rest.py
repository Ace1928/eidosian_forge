from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_nfs_service_rest(self):
    api = 'protocols/nfs/services'
    body = {'svm.name': self.parameters['vserver']}
    body.update(self.create_modify_body(body))
    dummy, error = rest_generic.post_async(self.rest_api, api, body, job_timeout=120)
    if error:
        self.module.fail_json(msg='Error creating nfs service for SVM %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())