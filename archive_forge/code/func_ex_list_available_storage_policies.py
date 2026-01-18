from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def ex_list_available_storage_policies(self, target):
    """
        Returns a list of available backup storage policies

        :param  target: The backup target to list available policies for
        :type   target: :class:`BackupTarget` or ``str``

        :rtype: ``list`` of :class:`DimensionDataBackupStoragePolicy`
        """
    server_id = self._target_to_target_address(target)
    response = self.connection.request_with_orgId_api_1('server/%s/backup/client/storagePolicy' % server_id, method='GET').object
    return self._to_storage_policies(response)