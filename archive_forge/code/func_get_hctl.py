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
def get_hctl(self, session, lun):
    """Given an iSCSI session return the host, channel, target, and lun."""
    glob_str = '/sys/class/iscsi_host/host*/device/session' + session
    paths = glob.glob(glob_str + '/target*')
    if paths:
        __, channel, target = os.path.split(paths[0])[1].split(':')
    else:
        target = channel = '-'
        paths = glob.glob(glob_str)
    if not paths:
        LOG.debug('No hctl found on session %s with lun %s', session, lun)
        return None
    host = paths[0][26:paths[0].index('/', 26)]
    res = (host, channel, target, lun)
    LOG.debug('HCTL %s found on session %s with lun %s', res, session, lun)
    return res