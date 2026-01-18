from openstack.accelerator.v2 import accelerator_request as _arq
from openstack.accelerator.v2 import deployable as _deployable
from openstack.accelerator.v2 import device as _device
from openstack.accelerator.v2 import device_profile as _device_profile
from openstack import proxy
def get_device_profile(self, uuid, fields=None):
    """Get a single device profile.

        :param uuid: The value can be the UUID of a device profile.
        :returns: One :class:
            `~openstack.accelerator.v2.device_profile.DeviceProfile`
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            device profile matching the criteria could be found.
        """
    return self._get(_device_profile.DeviceProfile, uuid)