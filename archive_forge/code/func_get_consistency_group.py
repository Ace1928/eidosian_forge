from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def get_consistency_group(self):
    """Retrieve consistency groups and return information on the expected group."""
    existing_volumes = self.get_all_volumes_by_id()
    if not self.cache['get_consistency_group']:
        try:
            rc, consistency_groups = self.request('storage-systems/%s/consistency-groups' % self.ssid)
            for consistency_group in consistency_groups:
                if consistency_group['label'] == self.group_name:
                    rc, member_volumes = self.request('storage-systems/%s/consistency-groups/%s/member-volumes' % (self.ssid, consistency_group['id']))
                    self.cache['get_consistency_group'].update({'consistency_group_id': consistency_group['cgRef'], 'alert_threshold_pct': consistency_group['fullWarnThreshold'], 'maximum_snapshots': consistency_group['autoDeleteLimit'], 'rollback_priority': consistency_group['rollbackPriority'], 'reserve_capacity_full_policy': consistency_group['repFullPolicy'], 'sequence_numbers': consistency_group['uniqueSequenceNumber'], 'base_volumes': []})
                    for member_volume in member_volumes:
                        base_volume = existing_volumes[member_volume['volumeId']]
                        base_volume_size_b = int(base_volume['totalSizeInBytes'])
                        total_reserve_capacity_b = int(member_volume['totalRepositoryCapacity'])
                        reserve_capacity_pct = int(round(float(total_reserve_capacity_b) / float(base_volume_size_b) * 100))
                        rc, concat = self.request('storage-systems/%s/repositories/concat/%s' % (self.ssid, member_volume['repositoryVolume']))
                        self.cache['get_consistency_group']['base_volumes'].append({'name': base_volume['name'], 'id': base_volume['id'], 'base_volume_size_b': base_volume_size_b, 'total_reserve_capacity_b': total_reserve_capacity_b, 'reserve_capacity_pct': reserve_capacity_pct, 'repository_volume_info': concat})
                    break
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve snapshot consistency groups! Error [%s]. Array [%s].' % (error, self.ssid))
    return self.cache['get_consistency_group']