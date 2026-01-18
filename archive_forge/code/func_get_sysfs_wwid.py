from __future__ import annotations
import glob
import os
import re
import time
from typing import Optional
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from os_brick import constants
from os_brick import exception
from os_brick import executor
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
def get_sysfs_wwid(self, device_names):
    """Return the wwid from sysfs in any of devices in udev format."""
    for device_name in device_names:
        try:
            with open('/sys/block/%s/device/wwid' % device_name) as f:
                wwid = f.read().strip()
        except IOError:
            continue
        udev_wwid = self.WWN_TYPES.get(wwid[:4], '8') + wwid[4:]
        return udev_wwid
    return ''