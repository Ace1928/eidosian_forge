from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def create_storage_pool(self):
    """Create new storage pool."""
    url = 'storage-systems/%s/symbol/createVolumeGroup?verboseErrorResponse=true' % self.ssid
    request_body = dict(label=self.name, candidate=self.get_candidate_drives())
    if self.raid_level == 'raidDiskPool':
        url = 'storage-systems/%s/symbol/createDiskPool?verboseErrorResponse=true' % self.ssid
        request_body.update(dict(backgroundOperationPriority='useDefault', criticalReconstructPriority='useDefault', degradedReconstructPriority='useDefault', poolUtilizationCriticalThreshold=65535, poolUtilizationWarningThreshold=0))
        if self.reserve_drive_count:
            request_body.update(dict(volumeCandidateData=dict(diskPoolVolumeCandidateData=dict(reconstructionReservedDriveCount=self.reserve_drive_count))))
    try:
        rc, resp = self.request(url, method='POST', data=request_body)
    except Exception as error:
        self.module.fail_json(msg='Failed to create storage pool. Array id [%s].  Error[%s].' % (self.ssid, to_native(error)))
    self.pool_detail = self.storage_pool