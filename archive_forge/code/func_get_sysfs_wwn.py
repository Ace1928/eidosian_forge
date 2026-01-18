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
def get_sysfs_wwn(self, device_names: list[str], mpath=None) -> str:
    """Return the wwid from sysfs in any of devices in udev format."""
    if mpath:
        try:
            with open('/sys/block/%s/dm/uuid' % mpath) as f:
                wwid = f.read().strip()[6:]
                if wwid:
                    return wwid
        except Exception as exc:
            LOG.warning('Failed to read the DM uuid: %s', exc)
    wwid = self.get_sysfs_wwid(device_names)
    glob_str = '/dev/disk/by-id/scsi-'
    wwn_paths = glob.glob(glob_str + '*')
    if wwid and glob_str + wwid in wwn_paths:
        return wwid
    device_names_set = set(device_names)
    for wwn_path in wwn_paths:
        try:
            if os.path.islink(wwn_path) and os.stat(wwn_path):
                path = os.path.realpath(wwn_path)
                if path.startswith('/dev/'):
                    name = path[5:]
                    if name.startswith('dm-'):
                        slaves_path = '/sys/class/block/%s/slaves' % name
                        dm_devs = os.listdir(slaves_path)
                        if device_names_set.intersection(dm_devs):
                            break
                    elif name in device_names_set:
                        break
        except OSError:
            continue
    else:
        return ''
    return wwn_path[len(glob_str):]