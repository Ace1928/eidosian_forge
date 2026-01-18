from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_cg_rest(self):
    """
        Retrieve consistency group with the given CG name or list of volumes
        """
    api = '/application/consistency-groups'
    query = {'svm.name': self.parameters['vserver'], 'fields': 'svm.uuid,name,uuid,'}
    if self.parameters.get('consistency_group') is not None:
        query['name'] = self.parameters['consistency_group']
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            self.module.fail_json(msg='Error searching for consistency group %s: %s' % (self.parameters['consistency_group'], to_native(error)), exception=traceback.format_exc())
        if record:
            self.cg_uuid = record.get('uuid')
    if self.parameters.get('volumes') is not None:
        query['fields'] += 'volumes.name,'
        records, error = rest_generic.get_0_or_more_records(self.rest_api, api, query)
        if error:
            self.module.fail_json(msg='Error searching for consistency group having volumes %s: %s' % (self.parameters['volumes'], to_native(error)), exception=traceback.format_exc())
        if records:
            for record in records:
                if record.get('volumes') is not None:
                    cg_volumes = [vol_item['name'] for vol_item in record['volumes']]
                    if cg_volumes == self.parameters['volumes']:
                        self.cg_uuid = record.get('uuid')
                        break
    return None