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
@mock.patch('glob.glob', side_effect=[[], ['/sys/block/sda/holders/dm-9']])
def test_find_sysfs_multipath_dm(self, glob_mock):
    device_names = ('sda', 'sdb')
    res = self.linuxscsi.find_sysfs_multipath_dm(device_names)
    self.assertEqual('dm-9', res)
    glob_mock.assert_has_calls([mock.call('/sys/block/sda/holders/dm-*'), mock.call('/sys/block/sdb/holders/dm-*')])