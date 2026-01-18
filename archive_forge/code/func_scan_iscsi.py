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
def scan_iscsi(self, host, channel='-', target='-', lun='-'):
    """Send an iSCSI scan request given the host and optionally the ctl."""
    LOG.debug('Scanning host %(host)s c: %(channel)s, t: %(target)s, l: %(lun)s)', {'host': host, 'channel': channel, 'target': target, 'lun': lun})
    self.echo_scsi_command('/sys/class/scsi_host/host%s/scan' % host, '%(c)s %(t)s %(l)s' % {'c': channel, 't': target, 'l': lun})