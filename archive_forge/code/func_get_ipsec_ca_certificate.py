from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_ipsec_ca_certificate(self):
    """GET IPsec CA certificate record"""
    self.uuid = self.get_certificate_uuid()
    if self.uuid is None:
        if self.parameters['state'] == 'absent':
            return None
        svm_or_scope = self.parameters['svm'] if self.parameters.get('svm') else 'cluster'
        self.module.fail_json(msg='Error: certificate %s is not installed in %s' % (self.parameters['name'], svm_or_scope))
    api = 'security/ipsec/ca-certificates/%s' % self.uuid
    record, error = rest_generic.get_one_record(self.rest_api, api)
    if error:
        if "entry doesn't exist" in error:
            return None
        self.module.fail_json(msg='Error fetching security IPsec CA certificate %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    return record if record else None