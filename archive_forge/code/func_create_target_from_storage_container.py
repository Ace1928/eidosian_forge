from libcloud.common.base import BaseDriver, ConnectionUserAndKey
from libcloud.backup.types import BackupTargetType
def create_target_from_storage_container(self, container, type=BackupTargetType.OBJECT, extra=None):
    """
        Creates a new backup target from an existing storage container

        :param node: The Container to backup
        :type  node: ``Container``

        :param type: Backup target type (Physical, Virtual, ...).
        :type type: :class:`BackupTargetType`

        :param extra: (optional) Extra attributes (driver specific).
        :type extra: ``dict``

        :rtype: Instance of :class:`.BackupTarget`
        """
    return self.create_target(name=container.name, address=container.get_cdn_url(), type=type, extra=None)