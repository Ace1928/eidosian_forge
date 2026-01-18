from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_snapshot_policy_rest(self):
    """
        get details of the snapshot policy with rest API.
        """
    if not self.use_rest:
        return self.get_snapshot_policy()
    query = {'name': self.parameters['name']}
    if self.parameters.get('vserver'):
        query['svm.name'] = self.parameters['vserver']
        query['scope'] = 'svm'
    else:
        query['scope'] = 'cluster'
    api = 'storage/snapshot-policies'
    fields = 'enabled,svm.uuid,comment,copies.snapmirror_label,copies.count,copies.prefix,copies.schedule.name,scope'
    record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
    if error:
        self.module.fail_json(msg='Error on fetching snapshot policy: %s' % error)
    if record:
        current = {'enabled': record['enabled'], 'name': record['name'], 'uuid': record['uuid'], 'comment': record.get('comment', ''), 'count': [], 'prefix': [], 'schedule': [], 'snapmirror_label': []}
        if query['scope'] == 'svm':
            current['svm_name'] = record['svm']['name']
            current['svm_uuid'] = record['svm']['uuid']
        if record['copies']:
            for item in record['copies']:
                current['count'].append(item['count'])
                current['prefix'].append(item['prefix'])
                current['schedule'].append(item['schedule']['name'])
                current['snapmirror_label'].append(item['snapmirror_label'])
        return current
    return record