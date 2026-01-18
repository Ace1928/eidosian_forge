from openstack.placement.v1 import resource_class as _resource_class
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.placement.v1 import (
from openstack.placement.v1 import trait as _trait
from openstack import proxy
from openstack import resource
def create_resource_provider_inventory(self, resource_provider, resource_class, *, total, **attrs):
    """Create a new resource provider inventory from attributes

        :param resource_provider: Either the ID of a resource provider or a
            :class:`~openstack.placement.v1.resource_provider.ResourceProvider`
            instance.
        :param total: The actual amount of the resource that the provider can
            accommodate.
        :param attrs: Keyword arguments which will be used to create a
            :class:`~openstack.placement.v1.resource_provider_inventory.ResourceProviderInventory`,
            comprised of the properties on the ResourceProviderInventory class.

        :returns: The results of resource provider inventory creation
        :rtype: :class:`~openstack.placement.v1.resource_provider_inventory.ResourceProviderInventory`
        """
    resource_provider_id = resource.Resource._get_id(resource_provider)
    resource_class_name = resource.Resource._get_id(resource_class)
    return self._create(_resource_provider_inventory.ResourceProviderInventory, resource_provider_id=resource_provider_id, resource_class=resource_class_name, total=total, **attrs)