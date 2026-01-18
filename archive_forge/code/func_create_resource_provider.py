from openstack.placement.v1 import resource_class as _resource_class
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.placement.v1 import (
from openstack.placement.v1 import trait as _trait
from openstack import proxy
from openstack import resource
def create_resource_provider(self, **attrs):
    """Create a new resource provider from attributes.

        :param attrs: Keyword arguments which will be used to create a
            :class:`~openstack.placement.v1.resource_provider.ResourceProvider`,
            comprised of the properties on the ResourceProvider class.

        :returns: The results of resource provider creation
        :rtype: :class:`~openstack.placement.v1.resource_provider.ResourceProvider`
        """
    return self._create(_resource_provider.ResourceProvider, **attrs)