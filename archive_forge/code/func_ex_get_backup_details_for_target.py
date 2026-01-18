from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def ex_get_backup_details_for_target(self, target):
    """
        Returns a backup details object for a target

        :param  target: The backup target to get details for
        :type   target: :class:`BackupTarget` or ``str``

        :rtype: :class:`DimensionDataBackupDetails`
        """
    if not isinstance(target, BackupTarget):
        target = self.ex_get_target_by_id(target)
        if target is None:
            return
    response = self.connection.request_with_orgId_api_1('server/%s/backup' % target.address, method='GET').object
    return self._to_backup_details(response, target)