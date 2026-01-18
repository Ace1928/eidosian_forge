from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def ex_list_available_schedule_policies(self, target):
    """
        Returns a list of available backup schedule policies

        :param  target: The backup target to list available policies for
        :type   target: :class:`BackupTarget` or ``str``

        :rtype: ``list`` of :class:`DimensionDataBackupSchedulePolicy`
        """
    server_id = self._target_to_target_address(target)
    response = self.connection.request_with_orgId_api_1('server/%s/backup/client/schedulePolicy' % server_id, method='GET').object
    return self._to_schedule_policies(response)