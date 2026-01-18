from openstack.accelerator.v2 import accelerator_request as _arq
from openstack.accelerator.v2 import deployable as _deployable
from openstack.accelerator.v2 import device as _device
from openstack.accelerator.v2 import device_profile as _device_profile
from openstack import proxy
def create_accelerator_request(self, **attrs):
    """Create an ARQs for a single device profile.

        :param kwargs attrs: request body.
        :returns: The created accelerator request instance.
        """
    return self._create(_arq.AcceleratorRequest, **attrs)