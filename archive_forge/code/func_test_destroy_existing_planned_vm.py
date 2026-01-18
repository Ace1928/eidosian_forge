from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import migrationutils
@ddt.data({'planned_vm': None}, {'planned_vm': mock.sentinel.planned_vm})
@ddt.unpack
@mock.patch.object(migrationutils.MigrationUtils, '_destroy_planned_vm')
@mock.patch.object(migrationutils.MigrationUtils, '_get_planned_vm')
def test_destroy_existing_planned_vm(self, mock_get_planned_vm, mock_destroy_planned_vm, planned_vm):
    mock_get_planned_vm.return_value = planned_vm
    self._migrationutils.destroy_existing_planned_vm(mock.sentinel.vm_name)
    mock_get_planned_vm.assert_called_once_with(mock.sentinel.vm_name, self._migrationutils._compat_conn)
    if planned_vm:
        mock_destroy_planned_vm.assert_called_once_with(planned_vm)
    else:
        self.assertFalse(mock_destroy_planned_vm.called)