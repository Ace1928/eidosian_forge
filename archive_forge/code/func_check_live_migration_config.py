import platform
from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import migrationutils
from os_win.utils.compute import vmutils
def check_live_migration_config(self):
    migration_svc = self._compat_conn.Msvm_VirtualSystemMigrationService()[0]
    vsmssd = self._compat_conn.Msvm_VirtualSystemMigrationServiceSettingData()
    vsmssd = vsmssd[0]
    if not vsmssd.EnableVirtualSystemMigration:
        raise exceptions.HyperVException(_('Live migration is not enabled on this host'))
    if not migration_svc.MigrationServiceListenerIPAddressList:
        raise exceptions.HyperVException(_('Live migration networks are not configured on this host'))