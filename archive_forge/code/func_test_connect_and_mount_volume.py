import os
from unittest import mock
import ddt
from os_brick.initiator.windows import smbfs
from os_brick.remotefs import windows_remotefs
from os_brick.tests.windows import test_base
@ddt.data(True, False)
@mock.patch.object(smbfs.WindowsSMBFSConnector, '_get_disk_path')
@mock.patch.object(smbfs.WindowsSMBFSConnector, 'ensure_share_mounted')
def test_connect_and_mount_volume(self, read_only, mock_ensure_mounted, mock_get_disk_path):
    self._load_connector(expect_raw_disk=True)
    fake_conn_props = dict(access_mode='ro' if read_only else 'rw')
    self._vhdutils.get_virtual_disk_physical_path.return_value = mock.sentinel.raw_disk_path
    mock_get_disk_path.return_value = mock.sentinel.image_path
    device_info = self._connector.connect_volume(fake_conn_props)
    expected_info = dict(type='file', path=mock.sentinel.raw_disk_path)
    self.assertEqual(expected_info, device_info)
    self._vhdutils.attach_virtual_disk.assert_called_once_with(mock.sentinel.image_path, read_only=read_only)
    self._vhdutils.get_virtual_disk_physical_path.assert_called_once_with(mock.sentinel.image_path)
    get_dev_num = self._diskutils.get_device_number_from_device_name
    get_dev_num.assert_called_once_with(mock.sentinel.raw_disk_path)
    self._diskutils.set_disk_offline.assert_called_once_with(get_dev_num.return_value)