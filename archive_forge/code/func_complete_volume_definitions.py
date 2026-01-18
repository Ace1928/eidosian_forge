from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def complete_volume_definitions(self):
    """Determine the complete self.volumes structure."""
    group = self.get_consistency_group()
    if not self.volumes:
        for volume in group['base_volumes']:
            self.volumes.update({volume['name']: {'reserve_capacity_pct': self.reserve_capacity_pct, 'preferred_reserve_storage_pool': self.preferred_reserve_storage_pool, 'snapshot_volume_writable': self.view_writable, 'snapshot_volume_validate': self.view_validate, 'snapshot_volume_host': self.view_host, 'snapshot_volume_lun': None}})
    existing_storage_pools_by_id = self.get_all_storage_pools_by_id()
    existing_storage_pools_by_name = self.get_all_storage_pools_by_name()
    existing_volumes_by_name = self.get_all_volumes_by_name()
    existing_volumes_by_id = self.get_all_volumes_by_id()
    existing_mappings = self.get_mapping_by_id()
    existing_host_and_hostgroup_by_id = self.get_all_hosts_and_hostgroups_by_id()
    existing_host_and_hostgroup_by_name = self.get_all_hosts_and_hostgroups_by_name()
    for volume_name, volume_info in self.volumes.items():
        base_volume_storage_pool_id = existing_volumes_by_name[volume_name]['volumeGroupRef']
        base_volume_storage_pool_name = existing_storage_pools_by_id[base_volume_storage_pool_id]['name']
        if not volume_info['preferred_reserve_storage_pool']:
            volume_info['preferred_reserve_storage_pool'] = base_volume_storage_pool_name
        elif volume_info['preferred_reserve_storage_pool'] not in existing_storage_pools_by_name.keys():
            self.module.fail_json(msg='Preferred storage pool or volume group does not exist! Storage pool [%s]. Group [%s]. Array [%s].' % (volume_info['preferred_reserve_storage_pool'], self.group_name, self.ssid))
        if self.state == 'present' and self.type == 'view':
            view_info = self.get_consistency_group_view()
            if volume_info['snapshot_volume_host']:
                if volume_info['snapshot_volume_host'] not in existing_host_and_hostgroup_by_name:
                    self.module.fail_json(msg='Specified host or host group does not exist! Host [%s]. Group [%s]. Array [%s].' % (volume_info['snapshot_volume_host'], self.group_name, self.ssid))
                if not volume_info['snapshot_volume_lun']:
                    if view_info:
                        for snapshot_volume in view_info['snapshot_volumes']:
                            if snapshot_volume['listOfMappings']:
                                mapping = snapshot_volume['listOfMappings'][0]
                                if volume_name == existing_volumes_by_id[snapshot_volume['baseVol']]['name'] and volume_info['snapshot_volume_host'] == existing_host_and_hostgroup_by_id[mapping['mapRef']]['name']:
                                    volume_info['snapshot_volume_lun'] = mapping['lun']
                                    break
                        else:
                            host_id = existing_host_and_hostgroup_by_name[volume_info['snapshot_volume_host']]['id']
                            for next_lun in range(1, 100):
                                if host_id not in existing_mappings.keys():
                                    existing_mappings.update({host_id: {}})
                                if next_lun not in existing_mappings[host_id].keys():
                                    volume_info['snapshot_volume_lun'] = next_lun
                                    existing_mappings[host_id].update({next_lun: None})
                                    break