from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def substitute_certificate_uuids(self, params):
    if 'external' not in params:
        return
    certificate = self.na_helper.safe_get(params, ['external', 'client_certificate'])
    if certificate:
        params['external']['client_certificate'] = {'uuid': self.get_security_certificate_uuid_rest(certificate, 'client')}
    certificates = self.na_helper.safe_get(params, ['external', 'server_ca_certificates'])
    if certificates:
        params['external']['server_ca_certificates'] = [{'uuid': self.get_security_certificate_uuid_rest(certificate, 'server_ca')} for certificate in certificates]