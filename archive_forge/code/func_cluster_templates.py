from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack import proxy
def cluster_templates(self, **query):
    """Return a generator of cluster_templates

        :param kwargs query: Optional query parameters to be sent to limit
            the resources being returned.

        :returns: A generator of cluster_template objects
        :rtype:
            :class:`~openstack.container_infrastructure_management.v1.cluster_template.ClusterTemplate`
        """
    return self._list(_cluster_template.ClusterTemplate, **query)