from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def get_security_ipsec_policy(self):
    """
        Get security ipsec policy.
        """
    api = 'security/ipsec/policies'
    query = {'name': self.parameters['name'], 'fields': 'uuid,enabled,local_endpoint,local_identity,remote_identity,protocol,remote_endpoint,action'}
    if self.parameters.get('authentication_method'):
        query['fields'] += ',authentication_method'
    if self.parameters.get('certificate'):
        query['fields'] += ',certificate'
    if self.parameters.get('svm'):
        query['svm.name'] = self.parameters['svm']
    else:
        query['scope'] = 'cluster'
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    if error:
        self.module.fail_json(msg='Error fetching security ipsec policy %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    if record:
        self.uuid = record['uuid']
        return {'action': self.na_helper.safe_get(record, ['action']), 'authentication_method': self.na_helper.safe_get(record, ['authentication_method']), 'certificate': self.na_helper.safe_get(record, ['certificate', 'name']), 'enabled': self.na_helper.safe_get(record, ['enabled']), 'local_endpoint': self.na_helper.safe_get(record, ['local_endpoint']), 'local_identity': self.na_helper.safe_get(record, ['local_identity']), 'protocol': self.na_helper.safe_get(record, ['protocol']), 'remote_endpoint': self.na_helper.safe_get(record, ['remote_endpoint']), 'remote_identity': self.na_helper.safe_get(record, ['remote_identity'])}
    return None