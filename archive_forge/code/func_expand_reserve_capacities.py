from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def expand_reserve_capacities(self, reserve_volumes):
    """Expand base volume(s) reserve capacity."""
    for volume_name, volume_info in reserve_volumes.items():
        candidate = self.get_candidate(volume_name, volume_info)
        expand_request = {'repositoryRef': volume_info['reserve_volume_id'], 'expansionCandidate': candidate['candidate']['candidate']}
        try:
            rc, resp = self.request('/storage-systems/%s/repositories/concat/%s/expand' % (self.ssid, volume_info['reserve_volume_id']), method='POST', data=expand_request)
        except Exception as error:
            self.module.fail_json(msg='Failed to expand reserve capacity volume! Group [%s]. Error [%s]. Array [%s].' % (self.group_name, error, self.ssid))