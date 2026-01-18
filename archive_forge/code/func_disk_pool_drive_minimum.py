from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
@property
def disk_pool_drive_minimum(self):
    """Provide the storage array's minimum disk pool drive count."""
    rc, attr = self.request('storage-systems/%s/symbol/getSystemAttributeDefaults' % self.ssid, ignore_errors=True)
    if rc != 200 or 'minimumDriveCount' not in attr['defaults']['diskPoolDefaultAttributes'].keys() or attr['defaults']['diskPoolDefaultAttributes']['minimumDriveCount'] == 0:
        return self.DEFAULT_DISK_POOL_MINIMUM_DISK_COUNT
    return attr['defaults']['diskPoolDefaultAttributes']['minimumDriveCount']