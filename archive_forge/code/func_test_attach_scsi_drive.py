from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, 'get_free_controller_slot')
@mock.patch.object(vmutils.VMUtils, '_get_vm_scsi_controller')
def test_attach_scsi_drive(self, mock_get_vm_scsi_controller, mock_get_free_controller_slot):
    mock_vm = self._lookup_vm()
    mock_get_vm_scsi_controller.return_value = self._FAKE_CTRL_PATH
    mock_get_free_controller_slot.return_value = self._FAKE_DRIVE_ADDR
    with mock.patch.object(self._vmutils, 'attach_drive') as mock_attach_drive:
        self._vmutils.attach_scsi_drive(mock_vm, self._FAKE_PATH, constants.DISK)
        mock_get_vm_scsi_controller.assert_called_once_with(mock_vm)
        mock_get_free_controller_slot.assert_called_once_with(self._FAKE_CTRL_PATH)
        mock_attach_drive.assert_called_once_with(mock_vm, self._FAKE_PATH, self._FAKE_CTRL_PATH, self._FAKE_DRIVE_ADDR, constants.DISK)