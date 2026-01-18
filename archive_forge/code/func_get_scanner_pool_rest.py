from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def get_scanner_pool_rest(self):
    """
        Check to see if a scanner pool exist or not using REST
        :return: record if it exist, None if it does not
        """
    self.get_svm_uuid()
    api = 'protocols/vscan/%s/scanner-pools' % self.svm_uuid
    query = {'name': self.parameters.get('scanner_pool'), 'fields': 'servers,privileged_users,'}
    if self.parameters.get('scanner_policy') is not None:
        query['fields'] += 'role,'
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    if error:
        self.module.fail_json(msg='Error searching for Vscan Scanner Pool %s: %s' % (self.parameters['scanner_pool'], to_native(error)), exception=traceback.format_exc())
    if record:
        return {'scanner_pool': record.get('name'), 'hostnames': record.get('servers'), 'privileged_users': record.get('privileged_users'), 'scanner_policy': record.get('role')}
    return None