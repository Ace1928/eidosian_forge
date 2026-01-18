from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack import proxy
def find_cluster_template(self, name_or_id, ignore_missing=True):
    """Find a single cluster_template

        :param name_or_id: The name or ID of a cluster_template.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the resource does not exist.
            When set to ``True``, None will be returned when
            attempting to find a nonexistent resource.
        :returns: One
            :class:`~openstack.container_infrastructure_management.v1.cluster_template.ClusterTemplate`
            or None
        """
    return self._find(_cluster_template.ClusterTemplate, name_or_id, ignore_missing=ignore_missing)