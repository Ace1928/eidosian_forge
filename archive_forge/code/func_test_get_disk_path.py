import os
from unittest import mock
import ddt
from os_brick.initiator.windows import smbfs
from os_brick.remotefs import windows_remotefs
from os_brick.tests.windows import test_base
@ddt.data({}, {'mount_base': mock.sentinel.mount_base}, {'is_local_share': True}, {'is_local_share': True, 'local_path_for_loopbk': True})
@ddt.unpack
def test_get_disk_path(self, mount_base=None, local_path_for_loopbk=False, is_local_share=False):
    fake_mount_point = 'C:\\\\fake_mount_point'
    fake_share_name = 'fake_share'
    fake_local_share_path = 'C:\\%s' % fake_share_name
    fake_export_path = '\\\\host\\%s' % fake_share_name
    fake_disk_name = 'fake_disk.vhdx'
    fake_conn_props = dict(name=fake_disk_name, export=fake_export_path)
    self._remotefs.get_mount_base.return_value = mount_base
    self._remotefs.get_mount_point.return_value = fake_mount_point
    self._remotefs.get_local_share_path.return_value = fake_local_share_path
    self._remotefs.get_share_name.return_value = fake_share_name
    self._connector._local_path_for_loopback = local_path_for_loopbk
    self._connector._smbutils.is_local_share.return_value = is_local_share
    expecting_local = local_path_for_loopbk and is_local_share
    if mount_base:
        expected_export_path = fake_mount_point
    elif expecting_local:
        expected_export_path = fake_local_share_path
    else:
        expected_export_path = fake_export_path
    expected_disk_path = os.path.join(expected_export_path, fake_disk_name)
    disk_path = self._connector._get_disk_path(fake_conn_props)
    self.assertEqual(expected_disk_path, disk_path)
    if mount_base:
        self._remotefs.get_mount_point.assert_called_once_with(fake_export_path)
    elif expecting_local:
        self._connector._smbutils.is_local_share.assert_called_once_with(fake_export_path)
        self._remotefs.get_local_share_path.assert_called_once_with(fake_export_path)