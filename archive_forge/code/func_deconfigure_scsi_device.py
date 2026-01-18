from __future__ import annotations
import glob
import os
from typing import Iterable
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick.initiator import linuxscsi
def deconfigure_scsi_device(self, device_number, target_wwn, lun):
    """Write the LUN to the port's unit_remove attribute.

        If auto-discovery of LUNs is disabled on s390 platforms
        luns need to be removed from the configuration through the
        unit_remove interface
        """
    LOG.debug('Deconfigure lun for s390: device_number=%(device_num)s target_wwn=%(target_wwn)s target_lun=%(target_lun)s', {'device_num': device_number, 'target_wwn': target_wwn, 'target_lun': lun})
    zfcp_device_command = '/sys/bus/ccw/drivers/zfcp/%s/%s/unit_remove' % (device_number, target_wwn)
    LOG.debug('unit_remove call for s390 execute: %s', zfcp_device_command)
    try:
        self.echo_scsi_command(zfcp_device_command, lun)
    except putils.ProcessExecutionError as exc:
        LOG.warning('unit_remove call for s390 failed exit %(code)s, stderr %(stderr)s', {'code': exc.exit_code, 'stderr': exc.stderr})