import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def create_volume_backup(self, volume_id, name=None, description=None, force=False, wait=True, timeout=None, incremental=False, snapshot_id=None):
    """Create a volume backup.

        :param volume_id: the ID of the volume to backup.
        :param name: name of the backup, one will be generated if one is
            not provided
        :param description: description of the backup, one will be generated
            if one is not provided
        :param force: If set to True the backup will be created even if the
            volume is attached to an instance, if False it will not
        :param wait: If true, waits for volume backup to be created.
        :param timeout: Seconds to wait for volume backup creation. None is
            forever.
        :param incremental: If set to true, the backup will be incremental.
        :param snapshot_id: The UUID of the source snapshot to back up.

        :returns: The created volume ``Backup`` object.
        :raises: :class:`~openstack.exceptions.ResourceTimeout` if wait time
            exceeded.
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    payload = {'name': name, 'volume_id': volume_id, 'description': description, 'force': force, 'is_incremental': incremental, 'snapshot_id': snapshot_id}
    backup = self.block_storage.create_backup(**payload)
    if wait:
        backup = self.block_storage.wait_for_status(backup, wait=timeout)
    return backup