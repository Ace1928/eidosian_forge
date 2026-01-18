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
def find_sysfs_multipath_dm(self, device_names):
    """Find the dm device name given a list of device names

        :param device_names: Iterable with device names, not paths. ie: ['sda']
        :returns: String with the dm name or None if not found. ie: 'dm-0'
        """
    glob_str = '/sys/block/%s/holders/dm-*'
    for dev_name in device_names:
        dms = glob.glob(glob_str % dev_name)
        if dms:
            __, device_name, __, dm = dms[0].rsplit('/', 3)
            return dm
    return None