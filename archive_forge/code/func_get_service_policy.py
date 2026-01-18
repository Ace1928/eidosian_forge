from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_service_policy(self):
    api = 'network/ip/service-policies'
    query = {'name': self.parameters['name'], 'fields': 'name,uuid,ipspace,services,svm,scope'}
    if self.parameters.get('vserver') is None:
        query['scope'] = 'cluster'
    else:
        query['svm.name'] = self.parameters['vserver']
    if self.parameters.get('ipspace') is not None:
        query['ipspace.name'] = self.parameters['ipspace']
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    if error:
        msg = 'Error in get_service_policy: %s' % error
        self.module.fail_json(msg=msg)
    if record:
        return {'uuid': record['uuid'], 'name': record['name'], 'ipspace': record['ipspace']['name'], 'scope': record['scope'], 'vserver': self.na_helper.safe_get(record, ['svm', 'name']), 'services': record['services']}
    return None