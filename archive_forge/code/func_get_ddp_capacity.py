from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def get_ddp_capacity(self, expansion_drive_list):
    """Return the total usable capacity based on the additional drives."""

    def get_ddp_error_percent(_drive_count, _extent_count):
        """Determine the space reserved for reconstruction"""
        if _drive_count <= 36:
            if _extent_count <= 600:
                return 0.4
            elif _extent_count <= 1400:
                return 0.35
            elif _extent_count <= 6200:
                return 0.2
            elif _extent_count <= 50000:
                return 0.15
        elif _drive_count <= 64:
            if _extent_count <= 600:
                return 0.2
            elif _extent_count <= 1400:
                return 0.15
            elif _extent_count <= 6200:
                return 0.1
            elif _extent_count <= 50000:
                return 0.05
        elif _drive_count <= 480:
            if _extent_count <= 600:
                return 0.2
            elif _extent_count <= 1400:
                return 0.15
            elif _extent_count <= 6200:
                return 0.1
            elif _extent_count <= 50000:
                return 0.05
        self.module.fail_json(msg='Drive count exceeded the error percent table. Array[%s]' % self.ssid)

    def get_ddp_reserved_drive_count(_disk_count):
        """Determine the number of reserved drive."""
        reserve_count = 0
        if self.reserve_drive_count:
            reserve_count = self.reserve_drive_count
        elif _disk_count >= 256:
            reserve_count = 8
        elif _disk_count >= 192:
            reserve_count = 7
        elif _disk_count >= 128:
            reserve_count = 6
        elif _disk_count >= 64:
            reserve_count = 4
        elif _disk_count >= 32:
            reserve_count = 3
        elif _disk_count >= 12:
            reserve_count = 2
        elif _disk_count == 11:
            reserve_count = 1
        return reserve_count
    if self.pool_detail:
        drive_count = len(self.storage_pool_drives) + len(expansion_drive_list)
    else:
        drive_count = len(expansion_drive_list)
    drive_usable_capacity = min(min(self.get_available_drive_capacities()), min(self.get_available_drive_capacities(expansion_drive_list)))
    drive_data_extents = (drive_usable_capacity - 8053063680) / 536870912
    maximum_stripe_count = drive_count * drive_data_extents / 10
    error_percent = get_ddp_error_percent(drive_count, drive_data_extents)
    error_overhead = (drive_count * drive_data_extents / 10 * error_percent + 10) / 10
    total_stripe_count = maximum_stripe_count - error_overhead
    stripe_count_per_drive = total_stripe_count / drive_count
    reserved_stripe_count = get_ddp_reserved_drive_count(drive_count) * stripe_count_per_drive
    available_stripe_count = total_stripe_count - reserved_stripe_count
    return available_stripe_count * 4294967296