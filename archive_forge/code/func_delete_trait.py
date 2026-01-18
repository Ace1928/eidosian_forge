from openstack.placement.v1 import resource_class as _resource_class
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.placement.v1 import (
from openstack.placement.v1 import trait as _trait
from openstack import proxy
from openstack import resource
def delete_trait(self, trait, ignore_missing=True):
    """Delete a trait

        :param trait: The value can be either the ID of a trait or an
            :class:`~openstack.placement.v1.trait.Trait`, instance.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be raised when
            the resource provider inventory does not exist. When set to
            ``True``, no exception will be set when attempting to delete a
            nonexistent resource provider inventory.

        :returns: ``None``
        """
    self._delete(_trait.Trait, trait, ignore_missing=ignore_missing)