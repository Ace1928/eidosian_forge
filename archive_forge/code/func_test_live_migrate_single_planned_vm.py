import platform
from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import livemigrationutils
from os_win.utils.compute import vmutils
from os_win.utils import jobutils
@mock.patch.object(livemigrationutils.LiveMigrationUtils, '_live_migrate_vm')
@mock.patch.object(livemigrationutils.LiveMigrationUtils, '_get_vhd_setting_data')
@mock.patch.object(livemigrationutils.LiveMigrationUtils, '_get_planned_vm')
def test_live_migrate_single_planned_vm(self, mock_get_planned_vm, mock_get_vhd_sd, mock_live_migrate_vm):
    mock_vm = self._get_vm()
    mock_migr_svc = self._conn.Msvm_VirtualSystemMigrationService()[0]
    mock_migr_svc.MigrationServiceListenerIPAddressList = [mock.sentinel.FAKE_REMOTE_IP_ADDR]
    mock_get_planned_vm.return_value = mock_vm
    self.liveutils.live_migrate_vm(mock.sentinel.vm_name, mock.sentinel.FAKE_HOST)
    self.liveutils._live_migrate_vm.assert_called_once_with(self._conn, mock_vm, mock_vm, [mock.sentinel.FAKE_REMOTE_IP_ADDR], self.liveutils._get_vhd_setting_data.return_value, mock.sentinel.FAKE_HOST, self.liveutils._MIGRATION_TYPE_VIRTUAL_SYSTEM_AND_STORAGE)
    mock_get_planned_vm.assert_called_once_with(mock.sentinel.vm_name, self._conn)