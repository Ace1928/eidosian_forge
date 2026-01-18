from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack import proxy
def create_cluster_template(self, **attrs):
    """Create a new cluster_template from attributes

        :param dict attrs: Keyword arguments which will be used to create a
            :class:`~openstack.container_infrastructure_management.v1.cluster_template.ClusterTemplate`,
            comprised of the properties on the ClusterTemplate class.
        :returns: The results of cluster_template creation
        :rtype:
            :class:`~openstack.container_infrastructure_management.v1.cluster_template.ClusterTemplate`
        """
    return self._create(_cluster_template.ClusterTemplate, **attrs)