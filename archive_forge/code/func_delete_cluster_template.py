from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack import proxy
def delete_cluster_template(self, cluster_template, ignore_missing=True):
    """Delete a cluster_template

        :param cluster_template: The value can be either the ID of a
            cluster_template or a
            :class:`~openstack.container_infrastructure_management.v1.cluster_template.ClusterTemplate`
            instance.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be raised when
            the cluster_template does not exist. When set to ``True``, no
            exception will be set when attempting to delete a nonexistent
            cluster_template.
        :returns: ``None``
        """
    self._delete(_cluster_template.ClusterTemplate, cluster_template, ignore_missing=ignore_missing)