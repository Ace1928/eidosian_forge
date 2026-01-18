from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, 'attach_drive')
@mock.patch.object(vmutils.VMUtils, '_get_vm_ide_controller')
def test_attach_ide_drive(self, mock_get_ide_ctrl, mock_attach_drive):
    mock_vm = self._lookup_vm()
    self._vmutils.attach_ide_drive(self._FAKE_VM_NAME, self._FAKE_CTRL_PATH, self._FAKE_CTRL_ADDR, self._FAKE_DRIVE_ADDR)
    mock_get_ide_ctrl.assert_called_with(mock_vm, self._FAKE_CTRL_ADDR)
    mock_attach_drive.assert_called_once_with(self._FAKE_VM_NAME, self._FAKE_CTRL_PATH, mock_get_ide_ctrl.return_value, self._FAKE_DRIVE_ADDR, constants.DISK)