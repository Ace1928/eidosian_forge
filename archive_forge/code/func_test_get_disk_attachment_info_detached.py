from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_get_mounted_disk_resource_from_path')
def test_get_disk_attachment_info_detached(self, mock_get_disk_res):
    mock_get_disk_res.return_value = None
    self.assertRaises(exceptions.DiskNotFound, self._vmutils.get_disk_attachment_info, mock.sentinel.disk_path, mock.sentinel.is_physical, mock.sentinel.serial)
    mock_get_disk_res.assert_called_once_with(mock.sentinel.disk_path, mock.sentinel.is_physical, serial=mock.sentinel.serial)