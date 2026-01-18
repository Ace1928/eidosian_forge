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
@mock.patch.object(linuxscsi.LinuxSCSI, '_execute')
@mock.patch('os.path.exists', return_value=True)
def test_flush_device_io(self, exists_mock, exec_mock):
    device = '/dev/sda'
    self.linuxscsi.flush_device_io(device)
    exists_mock.assert_called_once_with(device)
    exec_mock.assert_called_once_with('blockdev', '--flushbufs', device, run_as_root=True, attempts=3, timeout=300, interval=10, root_helper=self.linuxscsi._root_helper)