from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_igroup_rest(self, name):
    api = 'protocols/san/igroups'
    fields = 'name,uuid,svm,initiators,os_type,protocol'
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 1):
        fields += ',igroups'
    query = dict(name=name, fields=fields)
    query['svm.name'] = self.parameters['vserver']
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    self.fail_on_error(error)
    if record:
        try:
            igroup_details = dict(name=record['name'], uuid=record['uuid'], vserver=record['svm']['name'], os_type=record['os_type'], initiator_group_type=record['protocol'], name_to_uuid=dict())
        except KeyError as exc:
            self.module.fail_json(msg='Error: unexpected igroup body: %s, KeyError on %s' % (str(record), str(exc)))
        igroup_details['name_to_key'] = {}
        for attr in ('igroups', 'initiators'):
            option = 'initiator_names' if attr == 'initiators' else attr
            if attr in record:
                igroup_details[option] = [item['name'] for item in record[attr]]
                if attr == 'initiators':
                    igroup_details['initiator_objects'] = [dict(name=item['name'], comment=item.get('comment')) for item in record[attr]]
                igroup_details['name_to_uuid'][option] = dict(((item['name'], item.get('uuid', item['name'])) for item in record[attr]))
            else:
                igroup_details[option] = []
                igroup_details['name_to_uuid'][option] = {}
        return igroup_details
    return None