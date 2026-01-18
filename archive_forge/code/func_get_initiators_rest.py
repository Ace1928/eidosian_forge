from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_initiators_rest(self):
    api = 'protocols/san/igroups'
    query = {'name': self.parameters['initiator_group'], 'svm.name': self.parameters['vserver']}
    fields = 'initiators,uuid'
    record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
    if error:
        self.module.fail_json(msg='Error fetching igroup info %s: %s' % (self.parameters['initiator_group'], error))
    current = []
    if record:
        self.uuid = record['uuid']
        if 'initiators' in record:
            current = [initiator['name'] for initiator in record['initiators']]
    return current