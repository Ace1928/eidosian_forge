from openstack.accelerator.v2 import accelerator_request as _arq
from openstack.accelerator.v2 import deployable as _deployable
from openstack.accelerator.v2 import device as _device
from openstack.accelerator.v2 import device_profile as _device_profile
from openstack import proxy
def delete_device_profile(self, device_profile, ignore_missing=True):
    """Delete a device profile

        :param device_profile: The value can be either the ID of a device
            profile or a
            :class:`~openstack.accelerator.v2.device_profile.DeviceProfile`
            instance.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the device profile does not exist.
            When set to ``True``, no exception will be set when
            attempting to delete a nonexistent device profile.
        :returns: ``None``
        """
    return self._delete(_device_profile.DeviceProfile, device_profile, ignore_missing=ignore_missing)