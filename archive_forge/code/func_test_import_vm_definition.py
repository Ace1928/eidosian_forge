from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import migrationutils
def test_import_vm_definition(self):
    mock_svc = self._migrationutils._vs_man_svc
    mock_svc.ImportSystemDefinition.return_value = (mock.sentinel.ref, mock.sentinel.job_path, mock.sentinel.ret_val)
    self._migrationutils.import_vm_definition(export_config_file_path=mock.sentinel.export_config_file_path, snapshot_folder_path=mock.sentinel.snapshot_folder_path)
    mock_svc.ImportSystemDefinition.assert_called_once_with(False, mock.sentinel.snapshot_folder_path, mock.sentinel.export_config_file_path)
    self._migrationutils._jobutils.check_ret_val.assert_called_once_with(mock.sentinel.ret_val, mock.sentinel.job_path)