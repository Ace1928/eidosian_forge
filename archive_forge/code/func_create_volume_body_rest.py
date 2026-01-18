from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def create_volume_body_rest(self):
    body = {'name': self.parameters['name'], 'svm.name': self.parameters['vserver']}
    if self.parameters.get('space_guarantee') is not None:
        body['guarantee.type'] = self.parameters['space_guarantee']
    body = self.aggregates_rest(body)
    if self.parameters.get('tags') is not None:
        body['_tags'] = self.parameters['tags']
    if self.parameters.get('size') is not None:
        body['size'] = self.parameters['size']
    if self.parameters.get('snapshot_policy') is not None:
        body['snapshot_policy.name'] = self.parameters['snapshot_policy']
    if self.parameters.get('unix_permissions') is not None:
        body['nas.unix_permissions'] = self.parameters['unix_permissions']
    if self.parameters.get('group_id') is not None:
        body['nas.gid'] = self.parameters['group_id']
    if self.parameters.get('user_id') is not None:
        body['nas.uid'] = self.parameters['user_id']
    if self.parameters.get('volume_security_style') is not None:
        body['nas.security_style'] = self.parameters['volume_security_style']
    if self.parameters.get('export_policy') is not None:
        body['nas.export_policy.name'] = self.parameters['export_policy']
    if self.parameters.get('junction_path') is not None:
        body['nas.path'] = self.parameters['junction_path']
    if self.parameters.get('comment') is not None:
        body['comment'] = self.parameters['comment']
    if self.parameters.get('type') is not None:
        body['type'] = self.parameters['type'].lower()
    if self.parameters.get('percent_snapshot_space') is not None:
        body['space.snapshot.reserve_percent'] = self.parameters['percent_snapshot_space']
    if self.parameters.get('language') is not None:
        body['language'] = self.parameters['language']
    if self.get_qos_policy_group() is not None:
        body['qos.policy.name'] = self.get_qos_policy_group()
    if self.parameters.get('tiering_policy') is not None:
        body['tiering.policy'] = self.parameters['tiering_policy']
    if self.parameters.get('encrypt') is not None:
        body['encryption.enabled'] = self.parameters['encrypt']
    if self.parameters.get('logical_space_enforcement') is not None:
        body['space.logical_space.enforcement'] = self.parameters['logical_space_enforcement']
    if self.parameters.get('logical_space_reporting') is not None:
        body['space.logical_space.reporting'] = self.parameters['logical_space_reporting']
    if self.parameters.get('tiering_minimum_cooling_days') is not None:
        body['tiering.min_cooling_days'] = self.parameters['tiering_minimum_cooling_days']
    if self.parameters.get('snaplock') is not None:
        body['snaplock'] = self.na_helper.filter_out_none_entries(self.parameters['snaplock'])
    if self.volume_style:
        body['style'] = self.volume_style
    if self.parameters.get('efficiency_policy') is not None:
        body['efficiency.policy.name'] = self.parameters['efficiency_policy']
    if self.get_compression():
        body['efficiency.compression'] = self.get_compression()
    if self.parameters.get('analytics'):
        body['analytics.state'] = self.parameters['analytics']
    body['state'] = self.bool_to_online(self.parameters['is_online'])
    return body