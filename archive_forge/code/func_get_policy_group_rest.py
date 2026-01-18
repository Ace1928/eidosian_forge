from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_policy_group_rest(self, policy_group_name):
    api = 'storage/qos/policies'
    query = {'name': policy_group_name, 'svm.name': self.parameters['vserver']}
    fields = 'name,svm'
    if 'fixed_qos_options' in self.parameters:
        fields += ',fixed'
    elif 'adaptive_qos_options' in self.parameters:
        fields += ',adaptive'
    record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
    if error:
        self.module.fail_json(msg='Error fetching qos policy group %s: %s' % (self.parameters['name'], error))
    current = None
    if record:
        self.uuid = record['uuid']
        current = {'name': record['name'], 'vserver': record['svm']['name']}
        if 'fixed' in record:
            current['fixed_qos_options'] = {}
            for fixed_qos_option in ['capacity_shared', 'max_throughput_iops', 'max_throughput_mbps', 'min_throughput_iops']:
                current['fixed_qos_options'][fixed_qos_option] = record['fixed'].get(fixed_qos_option)
            if self.na_helper.safe_get(self.parameters, ['fixed_qos_options', 'min_throughput_mbps']):
                current['fixed_qos_options']['min_throughput_mbps'] = record['fixed'].get('min_throughput_mbps')
        if 'adaptive' in record:
            current['adaptive_qos_options'] = {}
            for adaptive_qos_option in ['absolute_min_iops', 'expected_iops', 'peak_iops', 'block_size', 'expected_iops_allocation', 'peak_iops_allocation']:
                current['adaptive_qos_options'][adaptive_qos_option] = record['adaptive'].get(adaptive_qos_option)
    return current