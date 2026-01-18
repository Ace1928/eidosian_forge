from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import migrationutils
@mock.patch.object(migrationutils.MigrationUtils, '_get_export_setting_data')
def test_export_vm(self, mock_get_export_setting_data):
    mock_vm = self._migrationutils._vmutils._lookup_vm.return_value
    export_setting_data = mock_get_export_setting_data.return_value
    mock_svc = self._migrationutils._vs_man_svc
    mock_svc.ExportSystemDefinition.return_value = (mock.sentinel.job_path, mock.sentinel.ret_val)
    self._migrationutils.export_vm(vm_name=self._FAKE_VM_NAME, export_path=mock.sentinel.fake_export_path)
    self.assertEqual(constants.EXPORT_CONFIG_SNAPSHOTS_ALL, export_setting_data.CopySnapshotConfiguration)
    self.assertFalse(export_setting_data.CopyVmStorage)
    self.assertFalse(export_setting_data.CreateVmExportSubdirectory)
    mock_get_export_setting_data.assert_called_once_with(self._FAKE_VM_NAME)
    mock_svc.ExportSystemDefinition.assert_called_once_with(ComputerSystem=mock_vm.path_(), ExportDirectory=mock.sentinel.fake_export_path, ExportSettingData=export_setting_data.GetText_(1))
    self._migrationutils._jobutils.check_ret_val.assert_called_once_with(mock.sentinel.ret_val, mock.sentinel.job_path)