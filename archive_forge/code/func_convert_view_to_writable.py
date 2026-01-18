from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def convert_view_to_writable(self, convert_view_information_list):
    """Make consistency group point-in-time snapshot volumes writable."""
    for volume_name, volume_info in convert_view_information_list.items():
        candidate = self.get_candidate(volume_name, volume_info)
        convert_request = {'fullThreshold': self.alert_threshold_pct, 'repositoryCandidate': candidate['candidate']['candidate']}
        try:
            rc, convert = self.request('/storage-systems/%s/snapshot-volumes/%s/convertReadOnly' % (self.ssid, volume_info['snapshot_volume_id']), method='POST', data=convert_request)
        except Exception as error:
            self.module.fail_json(msg='Failed to convert snapshot volume to read/write! Snapshot volume [%s]. View [%s] Group [%s]. Array [%s]. Error [%s].' % (volume_info['snapshot_volume_id'], self.view_name, self.group_name, self.ssid, error))