from openstack.placement.v1 import resource_class as _resource_class
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.placement.v1 import (
from openstack.placement.v1 import trait as _trait
from openstack import proxy
from openstack import resource
def resource_classes(self, **query):
    """Retrieve a generator of resource classs.

        :param kwargs query: Optional query parameters to be sent to
            restrict the resource classs to be returned.

        :returns: A generator of resource class instances.
        """
    return self._list(_resource_class.ResourceClass, **query)