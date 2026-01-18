from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def delete_target(self, target):
    """
        Delete a backup target

        :param target: Backup target to delete
        :type  target: Instance of :class:`BackupTarget` or ``str``

        :rtype: ``bool``
        """
    server_id = self._target_to_target_address(target)
    response = self.connection.request_with_orgId_api_1('server/%s/backup?disable' % server_id, method='GET').object
    response_code = findtext(response, 'result', GENERAL_NS)
    return response_code in ['IN_PROGRESS', 'SUCCESS']