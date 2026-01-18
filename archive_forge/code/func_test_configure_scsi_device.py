import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
@mock.patch.object(os.path, 'exists', return_value=False)
def test_configure_scsi_device(self, mock_execute):
    device_number = '0.0.2319'
    target_wwn = '0x50014380242b9751'
    lun = 1
    self.lfc.configure_scsi_device(device_number, target_wwn, lun)
    expected_commands = ['tee -a /sys/bus/ccw/drivers/zfcp/0.0.2319/port_rescan', 'tee -a /sys/bus/ccw/drivers/zfcp/0.0.2319/0x50014380242b9751/unit_add']
    self.assertEqual(expected_commands, self.cmds)