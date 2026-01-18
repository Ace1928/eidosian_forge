from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def get_snapmirror_policy_rest(self):
    query = {'fields': 'uuid,name,svm.name,comment,network_compression_enabled,type,retention,identity_preservation,sync_type,transfer_schedule,', 'name': self.parameters['policy_name'], 'scope': self.scope}
    if self.scope == 'svm':
        query['svm.name'] = self.parameters['vserver']
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 10, 1):
        query['fields'] += 'copy_all_source_snapshots,'
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 11, 1):
        query['fields'] += 'copy_latest_source_snapshot,create_snapshot_on_source'
    api = 'snapmirror/policies'
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    if error:
        self.module.fail_json(msg='Error getting snapmirror policy: %s' % error)
    return self.format_record(record) if record else None