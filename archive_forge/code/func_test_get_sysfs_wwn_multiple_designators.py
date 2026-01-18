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
@mock.patch('os.listdir', return_value=['sda', 'sdd'])
@mock.patch('os.path.realpath', side_effect=('/other/path', '/dev/dm-5', '/dev/sda', '/dev/sdb'))
@mock.patch('os.path.islink', side_effect=(False,) + (True,) * 5)
@mock.patch('os.stat', side_effect=(False,) + (True,) * 4)
@mock.patch('glob.glob')
@mock.patch.object(linuxscsi.LinuxSCSI, 'get_sysfs_wwid')
def test_get_sysfs_wwn_multiple_designators(self, get_wwid_mock, glob_mock, stat_mock, islink_mock, realpath_mock, listdir_mock):
    glob_mock.return_value = ['/dev/disk/by-id/scsi-fail-link', '/dev/disk/by-id/scsi-fail-stat', '/dev/disk/by-id/scsi-non-dev', '/dev/disk/by-id/scsi-another-dm', '/dev/disk/by-id/scsi-wwid1', '/dev/disk/by-id/scsi-wwid2']
    get_wwid_mock.return_value = 'pre-wwid'
    devices = ['sdb', 'sdc']
    res = self.linuxscsi.get_sysfs_wwn(devices)
    self.assertEqual('wwid2', res)
    glob_mock.assert_called_once_with('/dev/disk/by-id/scsi-*')
    listdir_mock.assert_called_once_with('/sys/class/block/dm-5/slaves')
    get_wwid_mock.assert_called_once_with(devices)