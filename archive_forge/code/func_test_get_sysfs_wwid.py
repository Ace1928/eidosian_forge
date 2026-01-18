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
@ddt.data({'wwn_type': 't10.', 'num_val': '1'}, {'wwn_type': 'eui.', 'num_val': '2'}, {'wwn_type': 'naa.', 'num_val': '3'})
@ddt.unpack
@mock.patch('builtins.open')
def test_get_sysfs_wwid(self, open_mock, wwn_type, num_val):
    read_fail = mock.MagicMock()
    read_fail.__enter__.return_value.read.side_effect = IOError
    read_data = mock.MagicMock()
    read_data.__enter__.return_value.read.return_value = wwn_type + 'wwid1\n'
    open_mock.side_effect = (IOError, read_fail, read_data)
    res = self.linuxscsi.get_sysfs_wwid(['sda', 'sdb', 'sdc'])
    self.assertEqual(num_val + 'wwid1', res)
    open_mock.assert_has_calls([mock.call('/sys/block/sda/device/wwid'), mock.call('/sys/block/sdb/device/wwid'), mock.call('/sys/block/sdc/device/wwid')])