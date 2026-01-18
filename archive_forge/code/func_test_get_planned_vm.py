from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import migrationutils
@ddt.data([mock.sentinel.planned_vm], [])
def test_get_planned_vm(self, planned_vm):
    planned_computer_system = self._migrationutils._conn.Msvm_PlannedComputerSystem
    planned_computer_system.return_value = planned_vm
    actual_result = self._migrationutils._get_planned_vm(self._FAKE_VM_NAME, fail_if_not_found=False)
    if planned_vm:
        self.assertEqual(planned_vm[0], actual_result)
    else:
        self.assertIsNone(actual_result)
    planned_computer_system.assert_called_once_with(ElementName=self._FAKE_VM_NAME)