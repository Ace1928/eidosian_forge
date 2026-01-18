from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def map_view(self, map_information_list):
    """Map consistency group point-in-time snapshot volumes to host or host group."""
    existing_volumes = self.get_all_volumes_by_id()
    existing_host_or_hostgroups = self.get_all_hosts_and_hostgroups_by_id()
    for map_request in map_information_list:
        try:
            rc, mapping = self.request('storage-systems/%s/volume-mappings' % self.ssid, method='POST', data=map_request)
        except Exception as error:
            self.module.fail_json(msg='Failed to map snapshot volume! Snapshot volume [%s]. Target [%s]. Lun [%s]. Group [%s]. Array [%s]. Error [%s].' % (existing_volumes[map_request['mappableObjectId']], existing_host_or_hostgroups[map_request['targetId']], map_request['lun'], self.group_name, self.ssid, error))