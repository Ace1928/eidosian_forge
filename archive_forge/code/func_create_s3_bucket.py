from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_s3_bucket(self):
    api = 'protocols/s3/buckets'
    body = {'svm.name': self.parameters['vserver'], 'name': self.parameters['name']}
    body.update(self.form_create_or_modify_body())
    dummy, error = rest_generic.post_async(self.rest_api, api, body, job_timeout=120)
    if error:
        self.module.fail_json(msg='Error creating S3 bucket %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())