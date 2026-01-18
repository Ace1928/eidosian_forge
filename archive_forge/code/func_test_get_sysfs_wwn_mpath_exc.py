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
@mock.patch('builtins.open', side_effect=Exception)
@mock.patch('glob.glob')
@mock.patch.object(linuxscsi.LinuxSCSI, 'get_sysfs_wwid')
def test_get_sysfs_wwn_mpath_exc(self, get_wwid_mock, glob_mock, open_mock):
    glob_mock.return_value = ['/dev/disk/by-id/scsi-wwid1', '/dev/disk/by-id/scsi-wwid2']
    get_wwid_mock.return_value = 'wwid1'
    res = self.linuxscsi.get_sysfs_wwn(mock.sentinel.device_names, 'dm-1')
    open_mock.assert_called_once_with('/sys/block/dm-1/dm/uuid')
    self.assertEqual('wwid1', res)
    glob_mock.assert_called_once_with('/dev/disk/by-id/scsi-*')
    get_wwid_mock.assert_called_once_with(mock.sentinel.device_names)