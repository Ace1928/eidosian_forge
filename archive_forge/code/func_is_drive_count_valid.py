from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def is_drive_count_valid(self, drive_count):
    """Validate drive count criteria is met."""
    if self.criteria_drive_count and drive_count < self.criteria_drive_count:
        return False
    if self.raid_level == 'raidDiskPool':
        return drive_count >= self.disk_pool_drive_minimum
    if self.raid_level == 'raid0':
        return drive_count > 0
    if self.raid_level == 'raid1':
        return drive_count >= 2 and drive_count % 2 == 0
    if self.raid_level in ['raid3', 'raid5']:
        return 3 <= drive_count <= 30
    if self.raid_level == 'raid6':
        return 5 <= drive_count <= 30
    return False