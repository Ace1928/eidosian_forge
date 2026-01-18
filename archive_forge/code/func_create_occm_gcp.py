from __future__ import absolute_import, division, print_function
import uuid
import time
import base64
import json
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def create_occm_gcp(self):
    """
        Create Cloud Manager connector for GCP
        """
    if 'proxy_user_name' in self.parameters and 'proxy_url' not in self.parameters:
        self.module.fail_json(msg='Error: missing proxy_url')
    if 'proxy_password' in self.parameters and 'proxy_url' not in self.parameters:
        self.module.fail_json(msg='Error: missing proxy_url')
    proxy_certificates = []
    if 'proxy_certificates' in self.parameters:
        for c_file in self.parameters['proxy_certificates']:
            proxy_certificate, error = self.na_helper.encode_certificates(c_file)
            if error is not None:
                self.module.fail_json(msg='Error: not able to read certificate file %s' % c_file)
            proxy_certificates.append(proxy_certificate)
    self.parameters['region'] = self.parameters['zone'][:-2]
    response, client_id, error = self.deploy_gcp_vm(proxy_certificates)
    if error is not None:
        self.module.fail_json(msg='Error: create_occm_gcp: %s, %s' % (str(error), str(response)))
    return client_id