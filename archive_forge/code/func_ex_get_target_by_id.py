from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def ex_get_target_by_id(self, id):
    """
        Get a target by server id

        :param id: The id of the target you want to get
        :type  id: ``str``

        :rtype: :class:`BackupTarget`
        """
    node = self.connection.request_with_orgId_api_2('server/server/%s' % id).object
    return self._to_target(node)