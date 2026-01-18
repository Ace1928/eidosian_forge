from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_s3_service(self, modify):
    api = 'protocols/s3/services'
    body = {'name': self.parameters['name']}
    if modify.get('enabled') is not None:
        body['enabled'] = self.parameters['enabled']
    if modify.get('comment'):
        body['comment'] = self.parameters['comment']
    if modify.get('certificate_name'):
        body['certificate.name'] = self.parameters['certificate_name']
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.svm_uuid, body)
    if error:
        self.module.fail_json(msg='Error modifying S3 service %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())