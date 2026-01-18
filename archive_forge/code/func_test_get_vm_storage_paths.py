from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_get_vm_disks')
@mock.patch.object(vmutils.VMUtils, '_lookup_vm_check')
def test_get_vm_storage_paths(self, mock_lookup_vm_check, mock_get_vm_disks):
    mock_rasds = self._create_mock_disks()
    mock_get_vm_disks.return_value = ([mock_rasds[0]], [mock_rasds[1]])
    storage = self._vmutils.get_vm_storage_paths(self._FAKE_VM_NAME)
    disk_files, volume_drives = storage
    self.assertEqual([self._FAKE_VHD_PATH], disk_files)
    self.assertEqual([self._FAKE_VOLUME_DRIVE_PATH], volume_drives)
    mock_lookup_vm_check.assert_called_once_with(self._FAKE_VM_NAME)