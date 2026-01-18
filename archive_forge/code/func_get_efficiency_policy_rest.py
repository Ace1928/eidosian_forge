from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_efficiency_policy_rest(self):
    api = 'storage/volume-efficiency-policies'
    query = {'name': self.parameters['policy_name'], 'svm.name': self.parameters['vserver']}
    fields = 'name,type,start_threshold_percent,qos_policy,schedule,comment,duration,enabled'
    record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
    if error:
        self.module.fail_json(msg='Error searching for efficiency policy %s: %s' % (self.parameters['policy_name'], error))
    if record:
        self.uuid = record['uuid']
        current = {'policy_name': record['name'], 'policy_type': record['type'], 'qos_policy': record['qos_policy'], 'schedule': record['schedule']['name'] if 'schedule' in record else None, 'enabled': record['enabled'], 'duration': str(record['duration']) if 'duration' in record else None, 'changelog_threshold_percent': record['start_threshold_percent'] if 'start_threshold_percent' in record else None, 'comment': record['comment']}
        return current
    return None