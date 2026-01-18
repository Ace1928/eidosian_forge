from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def list_target_jobs(self, target):
    """
        List the backup jobs on a target

        :param target: Backup target with the backup data
        :type  target: Instance of :class:`BackupTarget`

        :rtype: ``list`` of :class:`BackupTargetJob`
        """
    raise NotImplementedError('list_target_jobs not implemented for this driver')