from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_security_ssh_rest(self):
    """
        Retrieves the SSH server configuration for the SVM or cluster.
        """
    fields = ['key_exchange_algorithms', 'ciphers', 'mac_algorithms', 'max_authentication_retry_count']
    query = {}
    if self.parameters.get('vserver'):
        api = 'security/ssh/svms'
        query['svm.name'] = self.parameters['vserver']
        fields.append('svm.uuid')
    else:
        api = 'security/ssh'
    query['fields'] = ','.join(fields)
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    if error:
        self.module.fail_json(msg=error)
    return record