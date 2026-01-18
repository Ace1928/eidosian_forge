import platform
from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import livemigrationutils
from os_win.utils.compute import vmutils
from os_win.utils import jobutils
def test_live_migrate_vm_helper(self):
    mock_conn_local = mock.MagicMock()
    mock_vm = mock.MagicMock()
    mock_vsmsd_cls = mock_conn_local.Msvm_VirtualSystemMigrationSettingData
    mock_vsmsd = mock_vsmsd_cls.return_value[0]
    mock_vsmsvc = mock_conn_local.Msvm_VirtualSystemMigrationService()[0]
    mock_vsmsvc.MigrateVirtualSystemToHost.return_value = (self._FAKE_RET_VAL, mock.sentinel.FAKE_JOB_PATH)
    self.liveutils._live_migrate_vm(mock_conn_local, mock_vm, None, [mock.sentinel.FAKE_REMOTE_IP_ADDR], mock.sentinel.FAKE_RASD_PATH, mock.sentinel.FAKE_HOST, mock.sentinel.migration_type)
    mock_vsmsd_cls.assert_called_once_with(MigrationType=mock.sentinel.migration_type)
    mock_vsmsvc.MigrateVirtualSystemToHost.assert_called_once_with(ComputerSystem=mock_vm.path_.return_value, DestinationHost=mock.sentinel.FAKE_HOST, MigrationSettingData=mock_vsmsd.GetText_.return_value, NewResourceSettingData=mock.sentinel.FAKE_RASD_PATH)