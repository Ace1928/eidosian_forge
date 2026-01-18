import platform
from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import livemigrationutils
from os_win.utils.compute import vmutils
from os_win.utils import jobutils
def test_create_planned_vm_helper(self):
    mock_vm = mock.MagicMock()
    mock_v2 = mock.MagicMock()
    mock_vsmsd_cls = mock_v2.Msvm_VirtualSystemMigrationSettingData
    mock_vsmsd = mock_vsmsd_cls.return_value[0]
    self._conn.Msvm_PlannedComputerSystem.return_value = [mock_vm]
    migr_svc = mock_v2.Msvm_VirtualSystemMigrationService()[0]
    migr_svc.MigrateVirtualSystemToHost.return_value = (self._FAKE_RET_VAL, mock.sentinel.FAKE_JOB_PATH)
    resulted_vm = self.liveutils._create_planned_vm(self._conn, mock_v2, mock_vm, [mock.sentinel.FAKE_REMOTE_IP_ADDR], mock.sentinel.FAKE_HOST)
    self.assertEqual(mock_vm, resulted_vm)
    mock_vsmsd_cls.assert_called_once_with(MigrationType=self.liveutils._MIGRATION_TYPE_STAGED)
    migr_svc.MigrateVirtualSystemToHost.assert_called_once_with(ComputerSystem=mock_vm.path_.return_value, DestinationHost=mock.sentinel.FAKE_HOST, MigrationSettingData=mock_vsmsd.GetText_.return_value)
    self.liveutils._jobutils.check_ret_val.assert_called_once_with(mock.sentinel.FAKE_JOB_PATH, self._FAKE_RET_VAL)