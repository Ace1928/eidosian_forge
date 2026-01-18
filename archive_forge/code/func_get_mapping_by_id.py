from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def get_mapping_by_id(self):
    """Retrieve and return a dictionary of """
    if not self.cache['get_mapping_by_id']:
        existing_hosts_and_hostgroups_by_id = self.get_all_hosts_and_hostgroups_by_id()
        existing_hosts_and_hostgroups_by_name = self.get_all_hosts_and_hostgroups_by_name()
        try:
            rc, mappings = self.request('storage-systems/%s/volume-mappings' % self.ssid)
            for mapping in mappings:
                host_ids = [mapping['mapRef']]
                map_entry = {mapping['lun']: mapping['volumeRef']}
                if mapping['type'] == 'cluster':
                    host_ids = existing_hosts_and_hostgroups_by_id[mapping['mapRef']]['host_ids']
                    if mapping['mapRef'] in self.cache['get_mapping_by_id'].keys():
                        self.cache['get_mapping_by_id'][mapping['mapRef']].update(map_entry)
                        self.cache['get_mapping_by_name'][mapping['mapRef']].update(map_entry)
                    else:
                        self.cache['get_mapping_by_id'].update({mapping['mapRef']: map_entry})
                        self.cache['get_mapping_by_name'].update({mapping['mapRef']: map_entry})
                for host_id in host_ids:
                    if host_id in self.cache['get_mapping_by_id'].keys():
                        self.cache['get_mapping_by_id'][mapping['mapRef']].update(map_entry)
                        self.cache['get_mapping_by_name'][mapping['mapRef']].update(map_entry)
                    else:
                        self.cache['get_mapping_by_id'].update({host_id: map_entry})
                        self.cache['get_mapping_by_name'].update({host_id: map_entry})
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve all volume map definitions! Error [%s]. Array [%s].' % (error, self.ssid))
    return self.cache['get_mapping_by_id']