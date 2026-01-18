from openstack.placement.v1 import resource_class as _resource_class
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.placement.v1 import (
from openstack.placement.v1 import trait as _trait
from openstack import proxy
from openstack import resource
def delete_resource_provider_inventory(self, resource_provider_inventory, resource_provider=None, ignore_missing=True):
    """Delete a resource provider inventory

        :param resource_provider_inventory: The value can be either the ID of a
            resource provider or an
            :class:`~openstack.placement.v1.resource_provider_inventory.ResourceProviderInventory`,
            instance.
        :param resource_provider: Either the ID of a resource provider or a
            :class:`~openstack.placement.v1.resource_provider.ResourceProvider`
            instance. This value must be specified when
            ``resource_provider_inventory`` is an ID.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be raised when
            the resource provider inventory does not exist. When set to
            ``True``, no exception will be set when attempting to delete a
            nonexistent resource provider inventory.

        :returns: ``None``
        """
    resource_provider_id = self._get_uri_attribute(resource_provider_inventory, resource_provider, 'resource_provider_id')
    self._delete(_resource_provider_inventory.ResourceProviderInventory, resource_provider_inventory, resource_provider_id=resource_provider_id, ignore_missing=ignore_missing)