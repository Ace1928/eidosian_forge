from openstack.cloud import _utils
from openstack import exceptions
def get_coe_cluster_certificate(self, cluster_id):
    """Get details about the CA certificate for a cluster by name or ID.

        :param cluster_id: ID of the cluster.

        :returns: Details about the CA certificate for the given cluster.
        """
    return self.container_infrastructure_management.get_cluster_certificate(cluster_id)