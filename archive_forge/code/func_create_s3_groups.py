from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def create_s3_groups(self):
    api = 'protocols/s3/services/%s/groups' % self.svm_uuid
    body = {'name': self.parameters['name'], 'users': self.parameters['users'], 'policies': self.parameters['policies']}
    if self.parameters.get('comment'):
        body['comment'] = self.parameters['comment']
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error creating S3 groups %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())