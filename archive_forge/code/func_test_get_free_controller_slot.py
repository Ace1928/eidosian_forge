from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, 'get_attached_disks')
def test_get_free_controller_slot(self, mock_get_attached_disks):
    mock_disk = mock.MagicMock()
    mock_disk.AddressOnParent = 3
    mock_get_attached_disks.return_value = [mock_disk]
    response = self._vmutils.get_free_controller_slot(self._FAKE_CTRL_PATH)
    mock_get_attached_disks.assert_called_once_with(self._FAKE_CTRL_PATH)
    self.assertEqual(response, 0)