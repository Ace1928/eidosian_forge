from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def get_snapmirror_policy(self):
    if self.use_rest:
        return self.get_snapmirror_policy_rest()
    snapmirror_policy_get_iter = netapp_utils.zapi.NaElement('snapmirror-policy-get-iter')
    snapmirror_policy_info = netapp_utils.zapi.NaElement('snapmirror-policy-info')
    snapmirror_policy_info.add_new_child('policy-name', self.parameters['policy_name'])
    snapmirror_policy_info.add_new_child('vserver', self.parameters['vserver'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(snapmirror_policy_info)
    snapmirror_policy_get_iter.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(snapmirror_policy_get_iter, True)
    except netapp_utils.zapi.NaApiError as error:
        if 'NetApp API failed. Reason - 13001:' in to_native(error):
            return None
        self.module.fail_json(msg='Error getting snapmirror policy %s: %s' % (self.parameters['policy_name'], to_native(error)), exception=traceback.format_exc())
    return_value = None
    if result and result.get_child_by_name('attributes-list'):
        snapmirror_policy_attributes = result['attributes-list']['snapmirror-policy-info']
        return_value = {'policy_name': snapmirror_policy_attributes['policy-name'], 'tries': snapmirror_policy_attributes['tries'], 'transfer_priority': snapmirror_policy_attributes['transfer-priority'], 'is_network_compression_enabled': self.na_helper.get_value_for_bool(True, snapmirror_policy_attributes['is-network-compression-enabled']), 'restart': snapmirror_policy_attributes['restart'], 'ignore_atime': self.na_helper.get_value_for_bool(True, snapmirror_policy_attributes['ignore-atime']), 'vserver': snapmirror_policy_attributes['vserver-name'], 'comment': '', 'snapmirror_label': [], 'keep': [], 'prefix': [], 'schedule': []}
        if snapmirror_policy_attributes.get_child_content('comment') is not None:
            return_value['comment'] = snapmirror_policy_attributes['comment']
        if snapmirror_policy_attributes.get_child_content('type') is not None:
            return_value['policy_type'] = snapmirror_policy_attributes['type']
        if snapmirror_policy_attributes.get_child_content('common-snapshot-schedule') is not None:
            return_value['common_snapshot_schedule'] = snapmirror_policy_attributes['common-snapshot-schedule']
        if snapmirror_policy_attributes.get_child_by_name('snapmirror-policy-rules'):
            for rule in snapmirror_policy_attributes['snapmirror-policy-rules'].get_children():
                if rule.get_child_content('snapmirror-label') in ['sm_created', 'all_source_snapshots']:
                    continue
                return_value['snapmirror_label'].append(rule.get_child_content('snapmirror-label'))
                return_value['keep'].append(int(rule.get_child_content('keep')))
                prefix = rule.get_child_content('prefix')
                if prefix is None or prefix == '-':
                    prefix = ''
                return_value['prefix'].append(prefix)
                schedule = rule.get_child_content('schedule')
                if schedule is None or schedule == '-':
                    schedule = ''
                return_value['schedule'].append(schedule)
    return return_value