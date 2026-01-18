from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_active_directory_rest(self):
    api = 'protocols/active-directory'
    query = {'name': self.parameters['account_name'], 'svm.name': self.parameters['vserver'], 'fields': 'fqdn,name,organizational_unit'}
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    if error:
        self.module.fail_json(msg='Error searching for Active Directory %s: %s' % (self.parameters['account_name'], to_native(error)), exception=traceback.format_exc())
    if record:
        self.svm_uuid = record['svm']['uuid']
        return {'account_name': record.get('name'), 'domain': record.get('fqdn'), 'organizational_unit': record.get('organizational_unit')}
    return None