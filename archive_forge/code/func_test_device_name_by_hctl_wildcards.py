import os
import os.path
import textwrap
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.initiator import linuxscsi
from os_brick.tests import base
@mock.patch('glob.glob')
def test_device_name_by_hctl_wildcards(self, glob_mock):
    glob_mock.return_value = ['/sys/class/scsi_host/host3/device/session1/target3:4:5/3:4:5:2/block/sda2', '/sys/class/scsi_host/host3/device/session1/target3:4:5/3:4:5:2/block/sda']
    res = self.linuxscsi.device_name_by_hctl('1', ('3', '-', '-', '2'))
    self.assertEqual('sda', res)
    glob_mock.assert_called_once_with('/sys/class/scsi_host/host3/device/session1/target3:*:*/3:*:*:2/block/*')