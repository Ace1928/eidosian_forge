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
def modify_volume_body_rest(self, params):
    body = {}
    for key, option, transform in [('analytics.state', 'analytics', None), ('guarantee.type', 'space_guarantee', None), ('space.snapshot.reserve_percent', 'percent_snapshot_space', None), ('snapshot_policy.name', 'snapshot_policy', None), ('nas.export_policy.name', 'export_policy', None), ('nas.unix_permissions', 'unix_permissions', None), ('nas.gid', 'group_id', None), ('nas.uid', 'user_id', None), ('qos.policy.name', 'qos_policy_group', None), ('qos.policy.name', 'qos_adaptive_policy_group', None), ('comment', 'comment', None), ('space.logical_space.enforcement', 'logical_space_enforcement', None), ('space.logical_space.reporting', 'logical_space_reporting', None), ('tiering.min_cooling_days', 'tiering_minimum_cooling_days', None), ('state', 'is_online', self.bool_to_online), ('_tags', 'tags', None), ('snapshot_directory_access_enabled', 'snapdir_access', None), ('access_time_enabled', 'atime_update', None), ('space.nearly_full_threshold_percent', 'vol_nearly_full_threshold_percent', None), ('space.full_threshold_percent', 'vol_full_threshold_percent', None)]:
        value = self.parameters.get(option)
        if value is not None and transform:
            value = transform(value)
        if value is not None:
            body[key] = value
    for key, option, transform in [('nas.security_style', 'volume_security_style', None), ('tiering.policy', 'tiering_policy', None), ('files.maximum', 'max_files', None)]:
        if params and params.get(option) is not None:
            body[key] = self.parameters[option]
    if params and params.get('snaplock') is not None:
        sl_dict = self.na_helper.filter_out_none_entries(self.parameters['snaplock']) or {}
        sl_dict.pop('type', None)
        if sl_dict:
            body['snaplock'] = sl_dict
    if params and params.get('snapshot_auto_delete') is not None:
        for key, option, transform in [('space.snapshot.autodelete.trigger', 'trigger', None), ('space.snapshot.autodelete.target_free_space', 'target_free_space', None), ('space.snapshot.autodelete.delete_order', 'delete_order', None), ('space.snapshot.autodelete.commitment', 'commitment', None), ('space.snapshot.autodelete.defer_delete', 'defer_delete', None), ('space.snapshot.autodelete.prefix', 'prefix', None), ('space.snapshot.autodelete.enabled', 'state', self.enabled_to_bool)]:
            if params and params['snapshot_auto_delete'].get(option) is not None:
                if transform:
                    body[key] = transform(self.parameters['snapshot_auto_delete'][option])
                else:
                    body[key] = self.parameters['snapshot_auto_delete'][option]
    return body