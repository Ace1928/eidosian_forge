from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import migrationutils
def test_destroy_planned_vm(self):
    mock_planned_vm = mock.MagicMock()
    mock_planned_vm.path_.return_value = mock.sentinel.planned_vm_path
    mock_vs_man_svc = self._migrationutils._vs_man_svc
    mock_vs_man_svc.DestroySystem.return_value = (mock.sentinel.job_path, mock.sentinel.ret_val)
    self._migrationutils._destroy_planned_vm(mock_planned_vm)
    mock_vs_man_svc.DestroySystem.assert_called_once_with(mock.sentinel.planned_vm_path)
    self._migrationutils._jobutils.check_ret_val.assert_called_once_with(mock.sentinel.ret_val, mock.sentinel.job_path)