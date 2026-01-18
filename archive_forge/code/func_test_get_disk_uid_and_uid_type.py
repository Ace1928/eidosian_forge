from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
@mock.patch.object(diskutils.DiskUtils, '_get_disk_by_number')
def test_get_disk_uid_and_uid_type(self, mock_get_disk):
    mock_disk = mock_get_disk.return_value
    uid, uid_type = self._diskutils.get_disk_uid_and_uid_type(mock.sentinel.disk_number)
    mock_get_disk.assert_called_once_with(mock.sentinel.disk_number)
    self.assertEqual(mock_disk.UniqueId, uid)
    self.assertEqual(mock_disk.UniqueIdFormat, uid_type)