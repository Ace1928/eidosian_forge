from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
@ddt.data({'disk_path': '\\\\?\\MPio#disk&ven_fakeVendor', 'expect_mpio': True}, {'disk_path': '\\\\?\\SCSI#disk&ven_fakeVendor', 'expect_mpio': False})
@ddt.unpack
@mock.patch.object(diskutils.DiskUtils, '_get_disk_by_number')
def test_is_mpio_disk(self, mock_get_disk, disk_path, expect_mpio):
    mock_disk = mock_get_disk.return_value
    mock_disk.Path = disk_path
    result = self._diskutils.is_mpio_disk(mock.sentinel.disk_number)
    self.assertEqual(expect_mpio, result)
    mock_get_disk.assert_called_once_with(mock.sentinel.disk_number)