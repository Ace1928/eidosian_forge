from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack import proxy
def create_cluster_certificate(self, **attrs):
    """Create a new cluster_certificate from CSR

        :param dict attrs: Keyword arguments which will be used to create a
            :class:`~openstack.container_infrastructure_management.v1.cluster_certificate.ClusterCertificate`,
            comprised of the properties on the ClusterCertificate class.
        :returns: The results of cluster_certificate creation
        :rtype:
            :class:`~openstack.container_infrastructure_management.v1.cluster_certificate.ClusterCertificate`
        """
    return self._create(_cluster_cert.ClusterCertificate, **attrs)