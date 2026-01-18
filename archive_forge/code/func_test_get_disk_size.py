from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
@mock.patch.object(diskutils.DiskUtils, '_get_disk_by_number')
def test_get_disk_size(self, mock_get_disk):
    disk_size = self._diskutils.get_disk_size(mock.sentinel.disk_number)
    self.assertEqual(mock_get_disk.return_value.Size, disk_size)
    mock_get_disk.assert_called_once_with(mock.sentinel.disk_number)