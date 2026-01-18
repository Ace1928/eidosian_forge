import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
def test_deconfigure_scsi_device(self):
    device_number = '0.0.2319'
    target_wwn = '0x50014380242b9751'
    lun = 1
    self.lfc.deconfigure_scsi_device(device_number, target_wwn, lun)
    expected_commands = ['tee -a /sys/bus/ccw/drivers/zfcp/0.0.2319/0x50014380242b9751/unit_remove']
    self.assertEqual(expected_commands, self.cmds)