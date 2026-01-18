from openstack.cloud import _utils
from openstack import exceptions
def delete_coe_cluster(self, name_or_id):
    """Delete a COE cluster.

        :param name_or_id: Name or unique ID of the cluster.

        :returns: True if the delete succeeded, False if the
            cluster was not found.
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    cluster = self.get_coe_cluster(name_or_id)
    if not cluster:
        self.log.debug('COE Cluster %(name_or_id)s does not exist', {'name_or_id': name_or_id}, exc_info=True)
        return False
    self.container_infrastructure_management.delete_cluster(cluster)
    return True