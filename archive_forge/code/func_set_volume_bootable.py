import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def set_volume_bootable(self, name_or_id, bootable=True):
    """Set a volume's bootable flag.

        :param name_or_id: Name or unique ID of the volume.
        :param bool bootable: Whether the volume should be bootable.
            (Defaults to True)

        :returns: None
        :raises: :class:`~openstack.exceptions.ResourceTimeout` if wait time
            exceeded.
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    volume = self.get_volume(name_or_id)
    if not volume:
        raise exceptions.SDKException('Volume {name_or_id} does not exist'.format(name_or_id=name_or_id))
    self.block_storage.set_volume_bootable_status(volume, bootable)