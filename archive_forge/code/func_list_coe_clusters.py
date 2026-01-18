from openstack.cloud import _utils
from openstack import exceptions
def list_coe_clusters(self):
    """List COE (Container Orchestration Engine) cluster.

        :returns: A list of container infrastructure management ``Cluster``
            objects.
        :raises: :class:`~openstack.exceptions.SDKException` if something goes
            wrong during the OpenStack API call.
        """
    return list(self.container_infrastructure_management.clusters())