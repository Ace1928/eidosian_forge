from openstack.placement.v1 import resource_class as _resource_class
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.placement.v1 import (
from openstack.placement.v1 import trait as _trait
from openstack import proxy
from openstack import resource
def get_resource_provider(self, resource_provider):
    """Get a single resource_provider.

        :param resource_provider: The value can be either the ID of a resource
            provider or an
            :class:`~openstack.placement.v1.resource_provider.ResourceProvider`,
            instance.

        :returns: An instance of
            :class:`~openstack.placement.v1.resource_provider.ResourceProvider`
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            resource provider matching the criteria could be found.
        """
    return self._get(_resource_provider.ResourceProvider, resource_provider)