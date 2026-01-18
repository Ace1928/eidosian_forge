from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def get_reserve_drive_count(self):
    """Retrieve the current number of reserve drives for raidDiskPool (Only for raidDiskPool)."""
    if not self.pool_detail:
        self.module.fail_json(msg='The storage pool must exist. Array [%s].' % self.ssid)
    if self.raid_level != 'raidDiskPool':
        self.module.fail_json(msg='The storage pool must be a raidDiskPool. Pool [%s]. Array [%s].' % (self.pool_detail['id'], self.ssid))
    return self.pool_detail['volumeGroupData']['diskPoolData']['reconstructionReservedDriveCount']