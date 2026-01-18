from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def create_scanner_pool_rest(self):
    """
        Create a Vscan Scanner Pool using REST
        :return: nothing
        """
    api = 'protocols/vscan/%s/scanner-pools' % self.svm_uuid
    body = {'name': self.parameters['scanner_pool'], 'servers': self.parameters['hostnames'], 'privileged_users': self.parameters['privileged_users']}
    if 'scanner_policy' in self.parameters:
        body['role'] = self.parameters['scanner_policy']
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error is not None:
        self.module.fail_json(msg='Error creating Vscan Scanner Pool %s: %s' % (self.parameters['scanner_pool'], to_native(error)), exception=traceback.format_exc())