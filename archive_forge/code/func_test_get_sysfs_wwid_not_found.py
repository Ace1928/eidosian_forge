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
@mock.patch('builtins.open', side_effect=IOError)
def test_get_sysfs_wwid_not_found(self, open_mock):
    res = self.linuxscsi.get_sysfs_wwid(['sda', 'sdb'])
    self.assertEqual('', res)
    open_mock.assert_has_calls([mock.call('/sys/block/sda/device/wwid'), mock.call('/sys/block/sdb/device/wwid')])