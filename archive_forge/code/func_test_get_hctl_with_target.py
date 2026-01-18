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
def test_get_hctl_with_target(self, glob_mock):
    glob_mock.return_value = ['/sys/class/iscsi_host/host3/device/session1/target3:4:5', '/sys/class/iscsi_host/host3/device/session1/target3:4:6']
    res = self.linuxscsi.get_hctl('1', '2')
    self.assertEqual(('3', '4', '5', '2'), res)
    glob_mock.assert_called_once_with('/sys/class/iscsi_host/host*/device/session1/target*')