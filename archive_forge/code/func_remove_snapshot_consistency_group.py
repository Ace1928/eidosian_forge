from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def remove_snapshot_consistency_group(self, info):
    """remove a new snapshot consistency group."""
    try:
        rc, resp = self.request('storage-systems/%s/consistency-groups/%s' % (self.ssid, info['consistency_group_id']), method='DELETE')
    except Exception as error:
        self.module.fail_json(msg='Failed to remove snapshot consistency group! Group [%s]. Array [%s].' % (self.group_name, self.ssid))