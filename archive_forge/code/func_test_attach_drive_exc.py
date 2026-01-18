from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_get_wmi_obj')
@mock.patch.object(vmutils.VMUtils, '_get_new_resource_setting_data')
def test_attach_drive_exc(self, mock_get_new_rsd, mock_get_wmi_obj):
    self._lookup_vm()
    mock_drive_res = mock.Mock()
    mock_disk_res = mock.Mock()
    mock_get_new_rsd.side_effect = [mock_drive_res, mock_disk_res]
    self._jobutils.add_virt_resource.side_effect = [[mock.sentinel.drive_res_path], exceptions.OSWinException]
    mock_get_wmi_obj.return_value = mock.sentinel.attached_drive_res
    self.assertRaises(exceptions.OSWinException, self._vmutils.attach_drive, mock.sentinel.vm_name, mock.sentinel.disk_path, mock.sentinel.ctrl_path, mock.sentinel.drive_addr, constants.DISK)
    mock_get_wmi_obj.assert_called_once_with(mock.sentinel.drive_res_path)
    self._jobutils.remove_virt_resource.assert_called_once_with(mock.sentinel.attached_drive_res)