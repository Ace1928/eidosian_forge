from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import migrationutils
def test_get_export_setting_data(self):
    mock_vm = self._migrationutils._vmutils._lookup_vm.return_value
    mock_conn = self._migrationutils._compat_conn
    mock_exp = mock_conn.Msvm_VirtualSystemExportSettingData
    mock_exp.return_value = [mock.sentinel.export_setting_data]
    expected_result = mock.sentinel.export_setting_data
    actual_result = self._migrationutils._get_export_setting_data(self._FAKE_VM_NAME)
    self.assertEqual(expected_result, actual_result)
    mock_exp.assert_called_once_with(InstanceID=mock_vm.InstanceID)