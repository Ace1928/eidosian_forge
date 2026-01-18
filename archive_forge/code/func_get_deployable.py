from openstack.accelerator.v2 import accelerator_request as _arq
from openstack.accelerator.v2 import deployable as _deployable
from openstack.accelerator.v2 import device as _device
from openstack.accelerator.v2 import device_profile as _device_profile
from openstack import proxy
def get_deployable(self, uuid, fields=None):
    """Get a single deployable.

        :param uuid: The value can be the UUID of a deployable.
        :returns: One :class:`~openstack.accelerator.v2.deployable.Deployable`
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            deployable matching the criteria could be found.
        """
    return self._get(_deployable.Deployable, uuid)