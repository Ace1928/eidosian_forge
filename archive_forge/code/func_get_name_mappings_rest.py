from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_name_mappings_rest(self, index=None):
    """
        Retrieves the name mapping configuration for SVM with rest API.
        """
    if index is None:
        index = self.parameters['index']
    query = {'svm.name': self.parameters.get('vserver'), 'index': index, 'direction': self.parameters.get('direction'), 'fields': 'svm.uuid,client_match,direction,index,pattern,replacement,'}
    api = 'name-services/name-mappings'
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    if error:
        self.module.fail_json(msg=error)
    if record:
        self.svm_uuid = record['svm']['uuid']
        return {'pattern': self.na_helper.safe_get(record, ['pattern']), 'direction': self.na_helper.safe_get(record, ['direction']), 'replacement': self.na_helper.safe_get(record, ['replacement']), 'client_match': record.get('client_match', None)}
    return None