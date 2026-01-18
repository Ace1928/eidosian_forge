from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_public_keys(self):
    api = 'security/authentication/publickeys'
    query = {'account.name': self.parameters['account'], 'fields': 'account,owner,index,public_key,comment'}
    if self.parameters.get('vserver') is None:
        query['scope'] = 'cluster'
    else:
        query['owner.name'] = self.parameters['vserver']
    if self.parameters.get('index') is not None:
        query['index'] = self.parameters['index']
    response, error = self.rest_api.get(api, query)
    if self.parameters.get('index') is not None:
        record, error = rrh.check_for_0_or_1_records(api, response, error)
        records = [record]
    else:
        records, error = rrh.check_for_0_or_more_records(api, response, error)
    if error:
        msg = 'Error in get_public_key: %s' % error
        self.module.fail_json(msg=msg)
    if records is None or records == [None]:
        records = []
    return [dict([(k, v if k != 'account' else v['name']) for k, v in record.items()]) for record in records]