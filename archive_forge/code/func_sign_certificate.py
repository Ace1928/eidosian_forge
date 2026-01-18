from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def sign_certificate(self, uuid):
    """
        sign certificate
        :return: a dictionary with key "public_certificate"
        """
    api = 'security/certificates/%s/sign' % uuid
    body = {'signing_request': self.parameters['signing_request']}
    optional_keys = ['expiry_time', 'hash_function']
    for key in optional_keys:
        if self.parameters.get(key) is not None:
            body[key] = self.parameters[key]
    params = {'return_records': 'true'}
    message, error = self.rest_api.post(api, body, params)
    if error:
        self.module.fail_json(msg='Error signing certificate: %s' % error)
    return message