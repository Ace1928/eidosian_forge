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
def remove_scsi_device(self, device: str, force: bool=False, exc=None, flush: bool=True) -> None:
    """Removes a scsi device based upon /dev/sdX name."""
    path = '/sys/block/%s/device/delete' % device.replace('/dev/', '')
    if os.path.exists(path):
        exc = exception.ExceptionChainer() if exc is None else exc
        if flush:
            with exc.context(force, 'Flushing %s failed', device):
                self.flush_device_io(device)
        LOG.debug('Remove SCSI device %(device)s with %(path)s', {'device': device, 'path': path})
        with exc.context(force, 'Removing %s failed', device):
            self.echo_scsi_command(path, '1')