from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def convert_drives_list_into_drive_info_by_ids(self):
    """Determine drive identifiers base on provided drive list. Provide usable_ids list to select subset."""
    tray_by_ids = self.tray_by_ids()
    drives = []
    for usable_drive in self.usable_drives:
        tray, drawer, slot = usable_drive
        for drive in self.drives:
            drawer_slot = drawer * tray_by_ids[drive['physicalLocation']['trayRef']]['drawer_count'] + slot
            if drawer_slot == drive['physicalLocation']['slot'] and tray == tray_by_ids[drive['physicalLocation']['trayRef']]['tray_number']:
                if drive['available']:
                    drives.append(drive['id'])
                break
    return drives