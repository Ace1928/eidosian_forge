from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_nvme_rest(self):
    api = 'protocols/nvme/services'
    params = {'svm.name': self.parameters['vserver'], 'fields': 'enabled'}
    record, error = rest_generic.get_one_record(self.rest_api, api, params)
    if error:
        self.module.fail_json(msg='Error fetching nvme info for vserver: %s' % self.parameters['vserver'])
    if record:
        self.svm_uuid = record['svm']['uuid']
        record['status_admin'] = record.pop('enabled')
        return record
    return None