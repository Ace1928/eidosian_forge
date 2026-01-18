from openstack.cloud import _utils
from openstack import exceptions
def list_cluster_templates(self, detail=False):
    """List cluster templates.

        :param bool detail. Ignored. Included for backwards compat.
            ClusterTemplates are always returned with full details.

        :returns: a list of dicts containing the cluster template details.
        :raises: :class:`~openstack.exceptions.SDKException` if something goes
            wrong during the OpenStack API call.
        """
    return list(self.container_infrastructure_management.cluster_templates())