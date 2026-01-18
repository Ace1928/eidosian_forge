import platform
from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import livemigrationutils
from os_win.utils.compute import vmutils
from os_win.utils import jobutils
def test_check_live_migration_config(self):
    mock_migr_svc = self._conn.Msvm_VirtualSystemMigrationService.return_value[0]
    conn_vsmssd = self._conn.Msvm_VirtualSystemMigrationServiceSettingData
    vsmssd = mock.MagicMock()
    vsmssd.EnableVirtualSystemMigration = True
    conn_vsmssd.return_value = [vsmssd]
    mock_migr_svc.MigrationServiceListenerIPAdressList.return_value = [mock.sentinel.FAKE_HOST]
    self.liveutils.check_live_migration_config()
    conn_vsmssd.assert_called_once_with()
    self._conn.Msvm_VirtualSystemMigrationService.assert_called_once_with()