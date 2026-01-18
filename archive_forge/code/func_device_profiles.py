from openstack.accelerator.v2 import accelerator_request as _arq
from openstack.accelerator.v2 import deployable as _deployable
from openstack.accelerator.v2 import device as _device
from openstack.accelerator.v2 import device_profile as _device_profile
from openstack import proxy
def device_profiles(self, **query):
    """Retrieve a generator of device profiles.

        :param kwargs query: Optional query parameters to be sent to
            restrict the device profiles to be returned.
        :returns: A generator of device profile instances.
        """
    return self._list(_device_profile.DeviceProfile, **query)