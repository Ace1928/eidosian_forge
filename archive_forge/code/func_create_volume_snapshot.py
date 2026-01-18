import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
@_utils.valid_kwargs('name', 'display_name', 'description', 'display_description')
def create_volume_snapshot(self, volume_id, force=False, wait=True, timeout=None, **kwargs):
    """Create a volume.

        :param volume_id: the ID of the volume to snapshot.
        :param force: If set to True the snapshot will be created even if the
            volume is attached to an instance, if False it will not
        :param name: name of the snapshot, one will be generated if one is
            not provided
        :param description: description of the snapshot, one will be generated
            if one is not provided
        :param wait: If true, waits for volume snapshot to be created.
        :param timeout: Seconds to wait for volume snapshot creation. None is
            forever.

        :returns: The created volume ``Snapshot`` object.
        :raises: :class:`~openstack.exceptions.ResourceTimeout` if wait time
            exceeded.
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    kwargs = self._get_volume_kwargs(kwargs)
    payload = {'volume_id': volume_id, 'force': force}
    payload.update(kwargs)
    snapshot = self.block_storage.create_snapshot(**payload)
    if wait:
        snapshot = self.block_storage.wait_for_status(snapshot, wait=timeout)
    return snapshot