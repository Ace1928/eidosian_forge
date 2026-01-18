from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@ddt.data({'drive_path': '\\\\ADCONTROLLER\\root\\virtualization\\v2:Msvm_DiskDrive.CreationClassName="Msvm_DiskDrive",DeviceID="Microsoft:6344C73D-6FD6-4A74-8BE8-8EEAC2737369\\\\0\\\\0\\\\D",SystemCreationClassName="Msvm_ComputerSystem"', 'exp_phys_disk': True}, {'drive_path': 'some_image.vhdx', 'exp_phys_disk': False})
@ddt.unpack
@mock.patch.object(vmutils.VMUtils, '_get_mounted_disk_resource_from_path')
def test_drive_to_boot_source(self, mock_get_disk_res_from_path, drive_path, exp_phys_disk):
    mock_drive = mock.MagicMock()
    mock_drive.Parent = mock.sentinel.bssd
    mock_get_disk_res_from_path.return_value = mock_drive
    exp_rasd_path = mock_drive.path_.return_value if exp_phys_disk else mock_drive.Parent
    mock_same_element = mock.MagicMock()
    self._vmutils._conn.Msvm_LogicalIdentity.return_value = [mock.Mock(SameElement=mock_same_element)]
    ret = self._vmutils._drive_to_boot_source(drive_path)
    self._vmutils._conn.Msvm_LogicalIdentity.assert_called_once_with(SystemElement=exp_rasd_path)
    mock_get_disk_res_from_path.assert_called_once_with(drive_path, is_physical=exp_phys_disk)
    expected_path = mock_same_element.path_.return_value
    self.assertEqual(expected_path, ret)