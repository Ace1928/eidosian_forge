from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def recover_target_out_of_place(self, target, recovery_point, recovery_target, path=None):
    """
        Recover a backup target to a recovery point out-of-place

        :param target: Backup target with the backup data
        :type  target: Instance of :class:`BackupTarget`

        :param recovery_point: Backup target with the backup data
        :type  recovery_point: Instance of :class:`BackupTarget`

        :param recovery_target: Backup target with to recover the data to
        :type  recovery_target: Instance of :class:`BackupTarget`

        :param path: The part of the recovery point to recover (optional)
        :type  path: ``str``

        :rtype: Instance of :class:`BackupTargetJob`
        """
    raise NotImplementedError('recover_target_out_of_place not implemented for this driver')