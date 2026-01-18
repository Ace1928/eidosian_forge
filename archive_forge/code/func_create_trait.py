from openstack.placement.v1 import resource_class as _resource_class
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.placement.v1 import (
from openstack.placement.v1 import trait as _trait
from openstack import proxy
from openstack import resource
def create_trait(self, name):
    """Create a new trait

        :param name: The name of the new trait

        :returns: The results of trait creation
        :rtype: :class:`~openstack.placement.v1.trait.Trait`
        """
    return self._create(_trait.Trait, name=name)