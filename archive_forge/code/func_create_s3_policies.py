from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def create_s3_policies(self):
    api = 'protocols/s3/services/%s/policies' % self.svm_uuid
    body = {'name': self.parameters['name']}
    if self.parameters.get('comment'):
        body['comment'] = self.parameters['comment']
    if self.parameters.get('statements'):
        body['statements'] = self.parameters['statements']
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error creating S3 policy %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())